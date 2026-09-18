/* TensorRT V2DynamicExt/V3 plugin for shifted-window rotated sets. */

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>

namespace dynamic_bevfusion {
namespace {
using namespace nvinfer1;

constexpr char kName[] = "DSVTRotatedSet";
constexpr char kVersion[] = "1";
constexpr int kWindow = 30;
constexpr int kWindowsPerAxis = 13;
constexpr int kDenseWindows = 169;
constexpr int kWindowCells = 900;
constexpr int kSetSize = 90;
constexpr int kMaxSetsPerWindow = 10;
#ifndef DYNAMIC_BEVFUSION_MAX_SETS
#define DYNAMIC_BEVFUSION_MAX_SETS 512
#endif
#ifndef DYNAMIC_BEVFUSION_MAX_PILLARS
#define DYNAMIC_BEVFUSION_MAX_PILLARS 10000
#endif
constexpr int kMaxSets = DYNAMIC_BEVFUSION_MAX_SETS;
constexpr int kMaxPillars = DYNAMIC_BEVFUSION_MAX_PILLARS;
constexpr int kCoordCapacity = 360 * 360;
constexpr std::size_t kCubCapacity = 1U << 20;

template <typename T>
T* take(void*& cursor, std::size_t count) {
  auto address = reinterpret_cast<std::uintptr_t>(cursor);
  address = (address + 255U) & ~std::uintptr_t{255U};
  T* result = reinterpret_cast<T*>(address);
  cursor = reinterpret_cast<void*>(address + count * sizeof(T));
  return result;
}

__global__ void build_dense_metadata(
    const int* coords, const int* pillar_count, int shift, int* occupied,
    int* dense_counts, int* owner_y, int* owner_x, float* position_xy) {
  int pillar = blockIdx.x * blockDim.x + threadIdx.x;
  int pillars = pillar_count[0];
  if (pillar >= pillars || pillar >= kMaxPillars) return;
  int y = coords[pillar * 4 + 2];
  int x = coords[pillar * 4 + 3];
  int shifted_x = x + shift;
  int shifted_y = y + shift;
  int window_x = shifted_x / kWindow;
  int window_y = shifted_y / kWindow;
  int inner_x = shifted_x % kWindow;
  int inner_y = shifted_y % kWindow;
  int dense = window_x * kWindowsPerAxis + window_y;
  occupied[dense] = 1;
  atomicAdd(dense_counts + dense, 1);
  owner_y[dense * kWindowCells + inner_y * kWindow + inner_x] = pillar;
  owner_x[dense * kWindowCells + inner_x * kWindow + inner_y] = pillar;
  position_xy[pillar * 2 + 0] = static_cast<float>(inner_x) - 15.0F;
  position_xy[pillar * 2 + 1] = static_cast<float>(inner_y) - 15.0F;
}

__global__ void compact_axis_owners(int* owner_y, int* owner_x) {
  int dense = blockIdx.x;
  if (dense >= kDenseWindows || threadIdx.x != 0) return;
  int y_rank = 0;
  int x_rank = 0;
  for (int cell = 0; cell < kWindowCells; ++cell) {
    int y_pillar = owner_y[dense * kWindowCells + cell];
    if (y_pillar >= 0) owner_y[dense * kWindowCells + y_rank++] = y_pillar;
    int x_pillar = owner_x[dense * kWindowCells + cell];
    if (x_pillar >= 0) owner_x[dense * kWindowCells + x_rank++] = x_pillar;
  }
}

__global__ void compact_windows(const int* occupied, const int* dense_prefix,
                                const int* dense_counts, int* set_counts) {
  int dense = blockIdx.x * blockDim.x + threadIdx.x;
  if (dense >= kDenseWindows || !occupied[dense]) return;
  int contiguous = dense_prefix[dense];
  set_counts[contiguous] = (dense_counts[dense] + kSetSize - 1) / kSetSize;
}

__global__ void build_sets_dense(
    const int* occupied, const int* dense_prefix, const int* dense_counts,
    const int* set_counts, const int* set_offsets, const int* owner_y,
    const int* owner_x, int* indices, int* masks, int* gathers) {
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  int items_per_window = kMaxSetsPerWindow * kSetSize;
  if (item >= kDenseWindows * items_per_window) return;
  int dense = item / items_per_window;
  if (!occupied[dense]) return;
  int local = item % items_per_window;
  int local_set = local / kSetSize;
  int slot = local % kSetSize;
  int contiguous = dense_prefix[dense];
  int sets = set_counts[contiguous];
  if (local_set >= sets) return;
  int count = dense_counts[dense];
  int selected_rank = static_cast<int>(
      (static_cast<long long>(local_set * kSetSize + slot) * count) /
      (sets * kSetSize));
  int global_set = set_offsets[contiguous] + local_set;
  if (global_set >= kMaxSets) return;
  int flat_position = global_set * kSetSize + slot;
  int axis_pillars[2] = {
      owner_y[dense * kWindowCells + selected_rank],
      owner_x[dense * kWindowCells + selected_rank],
  };
#pragma unroll
  for (int axis = 0; axis < 2; ++axis) {
    int output = (axis * kMaxSets + global_set) * kSetSize + slot;
    int pillar = axis_pillars[axis];
    indices[output] = pillar;
    if (slot == 0) {
      masks[output] = 0;
    } else {
      int previous_rank = static_cast<int>(
          (static_cast<long long>(local_set * kSetSize + slot - 1) * count) /
          (sets * kSetSize));
      masks[output] = selected_rank == previous_rank;
    }
    if (pillar >= 0 && pillar < kMaxPillars) {
      atomicMax(gathers + axis * kMaxPillars + pillar, flat_position);
    }
  }
}

__global__ void unmask_first_slots(int* masks) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int entries = 2 * kMaxSets;
  if (index < entries) masks[index * kSetSize] = 0;
}

__global__ void write_total_sets(const int* occupied, const int* dense_prefix,
                                 const int* set_counts, const int* set_offsets,
                                 int* output, int shift_id) {
  if (blockIdx.x || threadIdx.x) return;
  int occupied_windows = dense_prefix[kDenseWindows - 1] +
                         occupied[kDenseWindows - 1];
  int total = 0;
  if (occupied_windows > 0) {
    int last = occupied_windows - 1;
    total = set_offsets[last] + set_counts[last];
  }
  output[shift_id] = total <= kMaxSets ? total : -total;
}

std::size_t plugin_workspace_size() {
  return 6 * 256 + 5 * kDenseWindows * sizeof(int) +
         2 * kDenseWindows * kWindowCells * sizeof(int) + kCubCapacity;
}

int32_t enqueue_plugin(PluginTensorDesc const* input_desc,
                       void const* const* inputs, void* const* outputs,
                       void* workspace, cudaStream_t stream) {
  if (input_desc[0].dims.nbDims != 2 ||
      input_desc[0].dims.d[0] != kCoordCapacity ||
      input_desc[0].dims.d[1] != 4 || input_desc[1].dims.nbDims != 1 ||
      input_desc[1].dims.d[0] != 1) {
    return 1;
  }
  int threads = 256;
  for (int shift_id = 0; shift_id < 2; ++shift_id) {
    int base = shift_id * 4;
    if (cudaMemsetAsync(outputs[base + 0], 0,
                        2 * kMaxSets * kSetSize * sizeof(int), stream) !=
            cudaSuccess ||
        cudaMemsetAsync(outputs[base + 1], 1,
                        2 * kMaxSets * kSetSize * sizeof(int), stream) !=
            cudaSuccess ||
        cudaMemsetAsync(outputs[base + 2], 0,
                        2 * kMaxPillars * sizeof(int), stream) != cudaSuccess ||
        cudaMemsetAsync(outputs[base + 3], 0,
                        kMaxPillars * 2 * sizeof(float), stream) !=
            cudaSuccess) {
      return 1;
    }
    unmask_first_slots<<<(2 * kMaxSets + threads - 1) / threads, threads, 0,
                         stream>>>(static_cast<int*>(outputs[base + 1]));

    void* cursor = workspace;
    int* occupied = take<int>(cursor, kDenseWindows);
    int* dense_prefix = take<int>(cursor, kDenseWindows);
    int* dense_counts = take<int>(cursor, kDenseWindows);
    int* set_counts = take<int>(cursor, kDenseWindows);
    int* set_offsets = take<int>(cursor, kDenseWindows);
    int* owner_y = take<int>(cursor, kDenseWindows * kWindowCells);
    int* owner_x = take<int>(cursor, kDenseWindows * kWindowCells);
    void* cub_storage = reinterpret_cast<void*>(
        (reinterpret_cast<std::uintptr_t>(cursor) + 255U) &
        ~std::uintptr_t{255U});
    if (cudaMemsetAsync(occupied, 0, kDenseWindows * sizeof(int), stream) !=
            cudaSuccess ||
        cudaMemsetAsync(dense_counts, 0, kDenseWindows * sizeof(int), stream) !=
            cudaSuccess ||
        cudaMemsetAsync(set_counts, 0, kDenseWindows * sizeof(int), stream) !=
            cudaSuccess ||
        cudaMemsetAsync(owner_y, 0xff,
                        kDenseWindows * kWindowCells * sizeof(int), stream) !=
            cudaSuccess ||
        cudaMemsetAsync(owner_x, 0xff,
                        kDenseWindows * kWindowCells * sizeof(int), stream) !=
            cudaSuccess) {
      return 1;
    }

    build_dense_metadata<<<(kMaxPillars + threads - 1) / threads, threads, 0,
                           stream>>>(
        static_cast<int const*>(inputs[0]), static_cast<int const*>(inputs[1]),
        shift_id * 15, occupied, dense_counts, owner_y, owner_x,
        static_cast<float*>(outputs[base + 3]));
    compact_axis_owners<<<kDenseWindows, 1, 0, stream>>>(owner_y, owner_x);
    std::size_t cub_bytes = kCubCapacity;
    if (cub::DeviceScan::ExclusiveSum(cub_storage, cub_bytes, occupied,
                                      dense_prefix, kDenseWindows, stream) !=
            cudaSuccess ||
        cub_bytes > kCubCapacity) {
      return 1;
    }
    compact_windows<<<1, threads, 0, stream>>>(
        occupied, dense_prefix, dense_counts, set_counts);
    cub_bytes = kCubCapacity;
    if (cub::DeviceScan::ExclusiveSum(cub_storage, cub_bytes, set_counts,
                                      set_offsets, kDenseWindows, stream) !=
            cudaSuccess ||
        cub_bytes > kCubCapacity) {
      return 1;
    }
    int work = kDenseWindows * kMaxSetsPerWindow * kSetSize;
    build_sets_dense<<<(work + threads - 1) / threads, threads, 0, stream>>>(
        occupied, dense_prefix, dense_counts, set_counts, set_offsets, owner_y,
        owner_x, static_cast<int*>(outputs[base + 0]),
        static_cast<int*>(outputs[base + 1]),
        static_cast<int*>(outputs[base + 2]));
    write_total_sets<<<1, 1, 0, stream>>>(
        occupied, dense_prefix, set_counts, set_offsets,
        static_cast<int*>(outputs[8]), shift_id);
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

#if NV_TENSORRT_MAJOR >= 10

class RotatedSetPlugin final : public IPluginV3,
                               public IPluginV3OneCore,
                               public IPluginV3OneBuild,
                               public IPluginV3OneRuntime {
 public:
  IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
    if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
    if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
    if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
    return nullptr;
  }
  IPluginV3* clone() noexcept override { return new (std::nothrow) RotatedSetPlugin(); }
  AsciiChar const* getPluginName() const noexcept override { return kName; }
  AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t,
                          DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }

  int32_t getOutputDataTypes(DataType* types, int32_t outputs,
                             DataType const*, int32_t) const noexcept override {
    if (outputs != 9) return 1;
    for (int output = 0; output < 9; ++output) types[output] = DataType::kINT32;
    types[3] = DataType::kFLOAT;
    types[7] = DataType::kFLOAT;
    return 0;
  }

  int32_t getOutputShapes(DimsExprs const*, int32_t inputs, DimsExprs const*,
                          int32_t, DimsExprs* out, int32_t outputs,
                          IExprBuilder& builder) noexcept override {
    if (inputs != 2 || outputs != 9) return 1;
    for (int shift = 0; shift < 2; ++shift) {
      int base = shift * 4;
      out[base + 0].nbDims = 3;
      out[base + 0].d[0] = builder.constant(2);
      out[base + 0].d[1] = builder.constant(kMaxSets);
      out[base + 0].d[2] = builder.constant(kSetSize);
      out[base + 1] = out[base + 0];
      out[base + 2].nbDims = 2;
      out[base + 2].d[0] = builder.constant(2);
      out[base + 2].d[1] = builder.constant(kMaxPillars);
      out[base + 3].nbDims = 2;
      out[base + 3].d[0] = builder.constant(kMaxPillars);
      out[base + 3].d[1] = builder.constant(2);
    }
    out[8].nbDims = 1;
    out[8].d[0] = builder.constant(2);
    return 0;
  }

  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io,
                                 int32_t inputs, int32_t outputs) noexcept override {
    if (inputs != 2 || outputs != 9 || pos < 0 || pos >= 11) return false;
    int output = pos - inputs;
    DataType expected = (output == 3 || output == 7) ? DataType::kFLOAT
                                                     : DataType::kINT32;
    if (pos < inputs) expected = DataType::kINT32;
    return io[pos].desc.format == TensorFormat::kLINEAR && io[pos].desc.type == expected;
  }
  int32_t getNbOutputs() const noexcept override { return 9; }

  size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t,
                          DynamicPluginTensorDesc const*, int32_t) const noexcept override {
    return plugin_workspace_size();
  }

  int32_t onShapeChange(PluginTensorDesc const* in, int32_t inputs,
                        PluginTensorDesc const*, int32_t outputs) noexcept override {
    return (inputs == 2 && outputs == 9 && in[0].dims.nbDims == 2 &&
            in[0].dims.d[0] == kCoordCapacity && in[0].dims.d[1] == 4 &&
            in[1].dims.nbDims == 1 && in[1].dims.d[0] == 1) ? 0 : 1;
  }

  int32_t enqueue(PluginTensorDesc const* input_desc, PluginTensorDesc const*,
                  void const* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    return enqueue_plugin(input_desc, inputs, outputs, workspace, stream);
  }

  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override {
    static PluginFieldCollection fields{0, nullptr};
    return &fields;
  }
};

class RotatedSetCreator final : public IPluginCreatorV3One {
 public:
  IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const*,
                          TensorRTPhase) noexcept override {
    return new (std::nothrow) RotatedSetPlugin();
  }
  PluginFieldCollection const* getFieldNames() noexcept override {
    static PluginFieldCollection fields{0, nullptr};
    return &fields;
  }
  AsciiChar const* getPluginName() const noexcept override { return kName; }
  AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }
};

#else

class RotatedSetPlugin final : public IPluginV2DynamicExt {
 public:
  int32_t getNbOutputs() const noexcept override { return 9; }
  DimsExprs getOutputDimensions(int32_t output_index, DimsExprs const*,
                                int32_t nb_inputs,
                                IExprBuilder& builder) noexcept override {
    DimsExprs output{};
    if (nb_inputs != 2 || output_index < 0 || output_index >= 9) return output;
    if (output_index == 8) {
      output.nbDims = 1;
      output.d[0] = builder.constant(2);
      return output;
    }
    int slot = output_index % 4;
    if (slot == 0 || slot == 1) {
      output.nbDims = 3;
      output.d[0] = builder.constant(2);
      output.d[1] = builder.constant(kMaxSets);
      output.d[2] = builder.constant(kSetSize);
    } else if (slot == 2) {
      output.nbDims = 2;
      output.d[0] = builder.constant(2);
      output.d[1] = builder.constant(kMaxPillars);
    } else {
      output.nbDims = 2;
      output.d[0] = builder.constant(kMaxPillars);
      output.d[1] = builder.constant(2);
    }
    return output;
  }
  bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* io,
                                 int32_t inputs,
                                 int32_t outputs) noexcept override {
    if (inputs != 2 || outputs != 9 || pos < 0 || pos >= 11) return false;
    int output = pos - inputs;
    DataType expected = (output == 3 || output == 7) ? DataType::kFLOAT
                                                     : DataType::kINT32;
    if (pos < inputs) expected = DataType::kINT32;
    return io[pos].format == TensorFormat::kLINEAR && io[pos].type == expected;
  }
  void configurePlugin(DynamicPluginTensorDesc const*, int32_t,
                       DynamicPluginTensorDesc const*,
                       int32_t) noexcept override {}
  size_t getWorkspaceSize(PluginTensorDesc const*, int32_t,
                          PluginTensorDesc const*,
                          int32_t) const noexcept override {
    return plugin_workspace_size();
  }
  int32_t enqueue(PluginTensorDesc const* input_desc, PluginTensorDesc const*,
                  void const* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    return enqueue_plugin(input_desc, inputs, outputs, workspace, stream);
  }
  DataType getOutputDataType(int32_t index, DataType const*,
                             int32_t) const noexcept override {
    return index == 3 || index == 7 ? DataType::kFLOAT : DataType::kINT32;
  }
  char const* getPluginType() const noexcept override { return kName; }
  char const* getPluginVersion() const noexcept override { return kVersion; }
  int32_t initialize() noexcept override { return 0; }
  void terminate() noexcept override {}
  size_t getSerializationSize() const noexcept override { return 0; }
  void serialize(void*) const noexcept override {}
  void destroy() noexcept override { delete this; }
  IPluginV2DynamicExt* clone() const noexcept override {
    auto* plugin = new (std::nothrow) RotatedSetPlugin();
    if (plugin != nullptr) plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }
  void setPluginNamespace(char const* plugin_namespace) noexcept override {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
  }
  char const* getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }

 private:
  std::string namespace_;
};

class RotatedSetCreator final : public IPluginCreator {
 public:
  char const* getPluginName() const noexcept override { return kName; }
  char const* getPluginVersion() const noexcept override { return kVersion; }
  PluginFieldCollection const* getFieldNames() noexcept override {
    static PluginFieldCollection fields{0, nullptr};
    return &fields;
  }
  IPluginV2* createPlugin(char const*,
                          PluginFieldCollection const*) noexcept override {
    auto* plugin = new (std::nothrow) RotatedSetPlugin();
    if (plugin != nullptr) plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
  }
  IPluginV2* deserializePlugin(char const*, void const*,
                               size_t serial_length) noexcept override {
    if (serial_length != 0) return nullptr;
    return createPlugin(nullptr, nullptr);
  }
  void setPluginNamespace(char const* plugin_namespace) noexcept override {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
  }
  char const* getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }

 private:
  std::string namespace_;
};

#endif
}  // namespace
}  // namespace dynamic_bevfusion

using DSVTRotatedSetCreator = dynamic_bevfusion::RotatedSetCreator;
REGISTER_TENSORRT_PLUGIN(DSVTRotatedSetCreator);
