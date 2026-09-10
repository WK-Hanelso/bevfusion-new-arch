/*
 * TensorRT 10 IPluginV3 for the non-learned DynamicPillarVFE boundary.
 *
 * Input:  points   float32 [N, 5]
 * Outputs:
 *   features       float32 [N, 11]      (invalid rows are zero)
 *   inverse        int32   [N]          (-1 for invalid rows)
 *   coords         int32   [129600, 4]  (valid prefix only)
 *   pillar_count   int32   [1]
 *
 * Pillars are ordered by the same sorted ``x * 360 + y`` key used by the
 * PyTorch integration.  The fixed coordinate capacity avoids a host sync or
 * a data-dependent output allocation inside TensorRT.
 */

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

namespace dynamic_bevfusion {
namespace {

using namespace nvinfer1;

constexpr char kPluginName[] = "DynamicPillarDecorate";
constexpr char kPluginVersion[] = "1";
constexpr int kPointChannels = 5;
constexpr int kFeatureChannels = 11;
constexpr int kGridX = 360;
constexpr int kGridY = 360;
constexpr int kCells = kGridX * kGridY;
#ifndef DYNAMIC_BEVFUSION_MAX_POINTS
#define DYNAMIC_BEVFUSION_MAX_POINTS 100000
#endif
constexpr int kMaxPoints = DYNAMIC_BEVFUSION_MAX_POINTS;
constexpr std::size_t kCubCapacity = 2U << 20;
constexpr float kMinX = -54.0F;
constexpr float kMinY = -54.0F;
constexpr float kMinZ = -5.0F;
constexpr float kMaxZ = 3.0F;
constexpr float kVoxelX = 0.3F;
constexpr float kVoxelY = 0.3F;
constexpr float kCenterZ = -1.0F;

std::size_t align256(std::size_t bytes) { return (bytes + 255U) & ~255U; }

template <typename T>
T* take(void*& cursor, std::size_t count) {
  auto address = reinterpret_cast<std::uintptr_t>(cursor);
  address = (address + 255U) & ~std::uintptr_t{255U};
  T* result = reinterpret_cast<T*>(address);
  cursor = reinterpret_cast<void*>(address + count * sizeof(T));
  return result;
}

__device__ int point_key(const float* point) {
  int x = static_cast<int>(floorf((point[0] - kMinX) / kVoxelX));
  int y = static_cast<int>(floorf((point[1] - kMinY) / kVoxelY));
  bool valid = x >= 0 && x < kGridX && y >= 0 && y < kGridY &&
               point[2] >= kMinZ && point[2] < kMaxZ;
  return valid ? x * kGridY + y : -1;
}

__global__ void accumulate_cells(const float* points, int count, int* point_keys,
                                 int* occupancy, float* xyz_sums,
                                 int* point_counts) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float* point = points + index * kPointChannels;
  int key = point_key(point);
  point_keys[index] = key;
  if (key < 0) return;
  atomicExch(occupancy + key, 1);
  atomicAdd(xyz_sums + key * 3 + 0, point[0]);
  atomicAdd(xyz_sums + key * 3 + 1, point[1]);
  atomicAdd(xyz_sums + key * 3 + 2, point[2]);
  atomicAdd(point_counts + key, 1);
}

__global__ void decorate_points(const float* points, const int* point_keys,
                                int count, const int* pillar_offsets,
                                const float* xyz_sums, const int* point_counts,
                                float* features, int* inverse) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  int key = point_keys[index];
  if (key < 0) {
    inverse[index] = -1;
    return;
  }
  inverse[index] = pillar_offsets[key];
  const float* point = points + index * kPointChannels;
  float* output = features + index * kFeatureChannels;
#pragma unroll
  for (int channel = 0; channel < kPointChannels; ++channel) {
    output[channel] = point[channel];
  }
  float reciprocal = 1.0F / static_cast<float>(point_counts[key]);
  output[5] = point[0] - xyz_sums[key * 3 + 0] * reciprocal;
  output[6] = point[1] - xyz_sums[key * 3 + 1] * reciprocal;
  output[7] = point[2] - xyz_sums[key * 3 + 2] * reciprocal;
  int x = key / kGridY;
  int y = key % kGridY;
  output[8] = point[0] - (x * kVoxelX + kVoxelX * 0.5F + kMinX);
  output[9] = point[1] - (y * kVoxelY + kVoxelY * 0.5F + kMinY);
  output[10] = point[2] - kCenterZ;
}

__global__ void emit_coords(const int* occupancy, const int* offsets,
                            int* coords, int* pillar_count) {
  int key = blockIdx.x * blockDim.x + threadIdx.x;
  if (key < kCells && occupancy[key]) {
    int pillar = offsets[key];
    int* output = coords + pillar * 4;
    output[0] = 0;
    output[1] = 0;
    output[2] = key % kGridY;
    output[3] = key / kGridY;
  }
  if (key == 0) {
    pillar_count[0] = offsets[kCells - 1] + occupancy[kCells - 1];
  }
}

class DynamicPillarDecoratePlugin final : public IPluginV3,
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

  IPluginV3* clone() noexcept override {
    return new (std::nothrow) DynamicPillarDecoratePlugin();
  }

  AsciiChar const* getPluginName() const noexcept override { return kPluginName; }
  AsciiChar const* getPluginVersion() const noexcept override { return kPluginVersion; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t,
                          DynamicPluginTensorDesc const*, int32_t) noexcept override {
    return 0;
  }

  int32_t getOutputDataTypes(DataType* output_types, int32_t nb_outputs,
                             DataType const*, int32_t) const noexcept override {
    if (nb_outputs != 4) return 1;
    output_types[0] = DataType::kFLOAT;
    output_types[1] = DataType::kINT32;
    output_types[2] = DataType::kINT32;
    output_types[3] = DataType::kINT32;
    return 0;
  }

  int32_t getOutputShapes(DimsExprs const* inputs, int32_t nb_inputs,
                          DimsExprs const*, int32_t, DimsExprs* outputs,
                          int32_t nb_outputs, IExprBuilder& builder) noexcept override {
    if (nb_inputs != 1 || nb_outputs != 4 || inputs[0].nbDims != 2) return 1;
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = builder.constant(kFeatureChannels);
    outputs[1].nbDims = 1;
    outputs[1].d[0] = inputs[0].d[0];
    outputs[2].nbDims = 2;
    outputs[2].d[0] = builder.constant(kCells);
    outputs[2].d[1] = builder.constant(4);
    outputs[3].nbDims = 1;
    outputs[3].d[0] = builder.constant(1);
    return 0;
  }

  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* in_out,
                                 int32_t nb_inputs, int32_t nb_outputs) noexcept override {
    if (nb_inputs != 1 || nb_outputs != 4 || pos < 0 || pos >= 5) return false;
    DataType expected = pos <= 1 ? DataType::kFLOAT : DataType::kINT32;
    return in_out[pos].desc.format == TensorFormat::kLINEAR &&
           in_out[pos].desc.type == expected;
  }

  int32_t getNbOutputs() const noexcept override { return 4; }

  size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t,
                          DynamicPluginTensorDesc const*, int32_t) const noexcept override {
    return align256(kMaxPoints * sizeof(int)) +
           3 * align256(kCells * sizeof(int)) +
           align256(kCells * 3 * sizeof(float)) + kCubCapacity + 256;
  }

  int32_t onShapeChange(PluginTensorDesc const* in, int32_t nb_inputs,
                        PluginTensorDesc const*, int32_t nb_outputs) noexcept override {
    if (nb_inputs != 1 || nb_outputs != 4 || in[0].dims.nbDims != 2 ||
        in[0].dims.d[0] < 0 || in[0].dims.d[0] > kMaxPoints ||
        in[0].dims.d[1] != kPointChannels) return 1;
    return 0;
  }

  int32_t enqueue(PluginTensorDesc const* input_desc,
                  PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void* workspace,
                  cudaStream_t stream) noexcept override {
    int count = input_desc[0].dims.d[0];
    if (count < 0 || count > kMaxPoints) return 1;
    void* cursor = workspace;
    int* point_keys = take<int>(cursor, kMaxPoints);
    int* occupancy = take<int>(cursor, kCells);
    int* point_counts = take<int>(cursor, kCells);
    int* offsets = take<int>(cursor, kCells);
    float* xyz_sums = take<float>(cursor, kCells * 3);
    void* cub_storage = reinterpret_cast<void*>((reinterpret_cast<std::uintptr_t>(cursor) + 255U) &
                                                ~std::uintptr_t{255U});

    if (cudaMemsetAsync(occupancy, 0, kCells * sizeof(int), stream) != cudaSuccess ||
        cudaMemsetAsync(point_counts, 0, kCells * sizeof(int), stream) != cudaSuccess ||
        cudaMemsetAsync(xyz_sums, 0, kCells * 3 * sizeof(float), stream) != cudaSuccess ||
        cudaMemsetAsync(outputs[0], 0,
                        static_cast<std::size_t>(count) * kFeatureChannels * sizeof(float),
                        stream) != cudaSuccess ||
        cudaMemsetAsync(outputs[2], 0, kCells * 4 * sizeof(int), stream) != cudaSuccess) {
      return 1;
    }
    int threads = 256;
    int point_blocks = (count + threads - 1) / threads;
    accumulate_cells<<<point_blocks, threads, 0, stream>>>(
        static_cast<float const*>(inputs[0]), count, point_keys, occupancy,
        xyz_sums, point_counts);
    std::size_t cub_bytes = kCubCapacity;
    cudaError_t status = cub::DeviceScan::ExclusiveSum(
        cub_storage, cub_bytes, occupancy, offsets, kCells, stream);
    if (status != cudaSuccess || cub_bytes > kCubCapacity) return 1;
    decorate_points<<<point_blocks, threads, 0, stream>>>(
        static_cast<float const*>(inputs[0]), point_keys, count, offsets,
        xyz_sums, point_counts, static_cast<float*>(outputs[0]),
        static_cast<int*>(outputs[1]));
    emit_coords<<<(kCells + threads - 1) / threads, threads, 0, stream>>>(
        occupancy, offsets, static_cast<int*>(outputs[2]),
        static_cast<int*>(outputs[3]));
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
  }

  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
    return clone();
  }

  PluginFieldCollection const* getFieldsToSerialize() noexcept override {
    static PluginFieldCollection fields{0, nullptr};
    return &fields;
  }
};

class DynamicPillarDecorateCreator final : public IPluginCreatorV3One {
 public:
  DynamicPillarDecorateCreator() { fields_.nbFields = 0; fields_.fields = nullptr; }
  IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const*,
                          TensorRTPhase) noexcept override {
    return new (std::nothrow) DynamicPillarDecoratePlugin();
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
  AsciiChar const* getPluginName() const noexcept override { return kPluginName; }
  AsciiChar const* getPluginVersion() const noexcept override { return kPluginVersion; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

 private:
  PluginFieldCollection fields_{};
};

}  // namespace
}  // namespace dynamic_bevfusion

using DynamicPillarDecorateCreator =
    dynamic_bevfusion::DynamicPillarDecorateCreator;
REGISTER_TENSORRT_PLUGIN(DynamicPillarDecorateCreator);
