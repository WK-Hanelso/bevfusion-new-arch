/*
 * TensorRT V2DynamicExt/V3 plugin for valid-prefix pillar-to-BEV scatter.
 *
 * Inputs:
 *   features      float/half [MAX_PILLARS, C]
 *   coords        int32      [129600, 4]
 *   pillar_count  int32      [1]
 * Output:
 *   dense_bev     float/half [1, C, 360, 360]
 *
 * Only the valid prefix [0, pillar_count) is written. Coordinates are unique,
 * so no reduction or atomic operation is required.
 */

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>

namespace dynamic_bevfusion {
namespace {

using namespace nvinfer1;

constexpr char kName[] = "DSVTDenseScatter";
constexpr char kVersion[] = "1";
#ifndef DYNAMIC_BEVFUSION_MAX_PILLARS
#define DYNAMIC_BEVFUSION_MAX_PILLARS 10000
#endif
constexpr int kMaxPillars = DYNAMIC_BEVFUSION_MAX_PILLARS;
constexpr int kCoordCapacity = 360 * 360;
constexpr int kGrid = 360;

template <typename T>
__global__ void scatter_valid_prefix(const T* features, const int* coords,
                                     const int* pillar_count, int channels,
                                     T* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int pillars = pillar_count[0];
  if (pillars < 0) return;
  if (pillars > kMaxPillars) pillars = kMaxPillars;
  int values = pillars * channels;
  if (index >= values) return;
  int pillar = index / channels;
  int channel = index % channels;
  int y = coords[pillar * 4 + 2];
  int x = coords[pillar * 4 + 3];
  if (x < 0 || x >= kGrid || y < 0 || y >= kGrid) return;
  output[(channel * kGrid + y) * kGrid + x] =
      features[pillar * channels + channel];
}

int32_t enqueue_plugin(PluginTensorDesc const* input_desc,
                       void const* const* inputs, void* const* outputs,
                       cudaStream_t stream) {
  if (input_desc[0].dims.nbDims != 2 ||
      input_desc[0].dims.d[0] != kMaxPillars ||
      input_desc[0].dims.d[1] <= 0 || input_desc[1].dims.nbDims != 2 ||
      input_desc[1].dims.d[0] != kCoordCapacity ||
      input_desc[1].dims.d[1] != 4 || input_desc[2].dims.nbDims != 1 ||
      input_desc[2].dims.d[0] != 1) {
    return 1;
  }
  int channels = input_desc[0].dims.d[1];
  std::size_t element_bytes =
      input_desc[0].type == DataType::kHALF ? sizeof(__half) : sizeof(float);
  std::size_t output_bytes =
      static_cast<std::size_t>(channels) * kGrid * kGrid * element_bytes;
  if (cudaMemsetAsync(outputs[0], 0, output_bytes, stream) != cudaSuccess) {
    return 1;
  }
  int values = kMaxPillars * channels;
  int blocks = (values + 255) / 256;
  if (input_desc[0].type == DataType::kFLOAT) {
    scatter_valid_prefix<<<blocks, 256, 0, stream>>>(
        static_cast<const float*>(inputs[0]),
        static_cast<const int*>(inputs[1]),
        static_cast<const int*>(inputs[2]), channels,
        static_cast<float*>(outputs[0]));
  } else if (input_desc[0].type == DataType::kHALF) {
    scatter_valid_prefix<<<blocks, 256, 0, stream>>>(
        static_cast<const __half*>(inputs[0]),
        static_cast<const int*>(inputs[1]),
        static_cast<const int*>(inputs[2]), channels,
        static_cast<__half*>(outputs[0]));
  } else {
    return 1;
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

#if NV_TENSORRT_MAJOR >= 10

class DenseScatterPlugin final : public IPluginV3,
                                 public IPluginV3OneCore,
                                 public IPluginV3OneBuild,
                                 public IPluginV3OneRuntime {
 public:
  IPluginCapability* getCapabilityInterface(
      PluginCapabilityType type) noexcept override {
    if (type == PluginCapabilityType::kCORE) {
      return static_cast<IPluginV3OneCore*>(this);
    }
    if (type == PluginCapabilityType::kBUILD) {
      return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME) {
      return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }

  IPluginV3* clone() noexcept override {
    return new (std::nothrow) DenseScatterPlugin();
  }
  AsciiChar const* getPluginName() const noexcept override { return kName; }
  AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t,
                          DynamicPluginTensorDesc const*,
                          int32_t) noexcept override {
    return 0;
  }

  int32_t getOutputDataTypes(DataType* output_types, int32_t nb_outputs,
                             DataType const* input_types,
                             int32_t nb_inputs) const noexcept override {
    if (nb_inputs != 3 || nb_outputs != 1) return 1;
    output_types[0] = input_types[0];
    return 0;
  }

  int32_t getOutputShapes(DimsExprs const* inputs, int32_t nb_inputs,
                          DimsExprs const*, int32_t, DimsExprs* outputs,
                          int32_t nb_outputs,
                          IExprBuilder& builder) noexcept override {
    if (nb_inputs != 3 || nb_outputs != 1 || inputs[0].nbDims != 2) return 1;
    outputs[0].nbDims = 4;
    outputs[0].d[0] = builder.constant(1);
    outputs[0].d[1] = inputs[0].d[1];
    outputs[0].d[2] = builder.constant(kGrid);
    outputs[0].d[3] = builder.constant(kGrid);
    return 0;
  }

  bool supportsFormatCombination(int32_t pos,
                                 DynamicPluginTensorDesc const* io,
                                 int32_t nb_inputs,
                                 int32_t nb_outputs) noexcept override {
    if (nb_inputs != 3 || nb_outputs != 1 || pos < 0 || pos >= 4) return false;
    DataType expected = DataType::kINT32;
    if (pos == 0 || pos == 3) expected = io[0].desc.type;
    bool feature_type = expected == DataType::kFLOAT || expected == DataType::kHALF;
    bool valid_type = pos == 0 || pos == 3 ? feature_type
                                           : expected == DataType::kINT32;
    return valid_type && io[pos].desc.format == TensorFormat::kLINEAR &&
           io[pos].desc.type == expected;
  }

  int32_t getNbOutputs() const noexcept override { return 1; }

  int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nb_inputs,
                        PluginTensorDesc const* outputs,
                        int32_t nb_outputs) noexcept override {
    if (nb_inputs != 3 || nb_outputs != 1 ||
        inputs[0].dims.nbDims != 2 ||
        inputs[0].dims.d[0] != kMaxPillars ||
        inputs[0].dims.d[1] <= 0 ||
        inputs[1].dims.nbDims != 2 ||
        inputs[1].dims.d[0] != kCoordCapacity ||
        inputs[1].dims.d[1] != 4 ||
        inputs[2].dims.nbDims != 1 ||
        inputs[2].dims.d[0] != 1 ||
        outputs[0].dims.nbDims != 4) {
      return 1;
    }
    return 0;
  }

  int32_t enqueue(PluginTensorDesc const* input_desc,
                  PluginTensorDesc const* output_desc,
                  void const* const* inputs,
                  void* const* outputs, void*,
                  cudaStream_t stream) noexcept override {
    return enqueue_plugin(input_desc, inputs, outputs, stream);
  }

  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
    return clone();
  }

  PluginFieldCollection const* getFieldsToSerialize() noexcept override {
    static PluginFieldCollection fields{0, nullptr};
    return &fields;
  }
};

class DenseScatterCreator final : public IPluginCreatorV3One {
 public:
  IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const*,
                          TensorRTPhase) noexcept override {
    return new (std::nothrow) DenseScatterPlugin();
  }
  PluginFieldCollection const* getFieldNames() noexcept override {
    static PluginFieldCollection fields{0, nullptr};
    return &fields;
  }
  AsciiChar const* getPluginName() const noexcept override { return kName; }
  AsciiChar const* getPluginVersion() const noexcept override {
    return kVersion;
  }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }
};

#else

class DenseScatterPlugin final : public IPluginV2DynamicExt {
 public:
  int32_t getNbOutputs() const noexcept override { return 1; }
  DimsExprs getOutputDimensions(int32_t output_index,
                                DimsExprs const* inputs, int32_t nb_inputs,
                                IExprBuilder& builder) noexcept override {
    DimsExprs output{};
    if (output_index != 0 || nb_inputs != 3 || inputs[0].nbDims != 2) {
      return output;
    }
    output.nbDims = 4;
    output.d[0] = builder.constant(1);
    output.d[1] = inputs[0].d[1];
    output.d[2] = builder.constant(kGrid);
    output.d[3] = builder.constant(kGrid);
    return output;
  }
  bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* io,
                                 int32_t nb_inputs,
                                 int32_t nb_outputs) noexcept override {
    if (nb_inputs != 3 || nb_outputs != 1 || pos < 0 || pos >= 4) return false;
    DataType expected = DataType::kINT32;
    if (pos == 0 || pos == 3) expected = io[0].type;
    bool feature_type = expected == DataType::kFLOAT || expected == DataType::kHALF;
    bool valid_type = (pos == 0 || pos == 3) ? feature_type
                                             : expected == DataType::kINT32;
    return valid_type && io[pos].format == TensorFormat::kLINEAR &&
           io[pos].type == expected;
  }
  void configurePlugin(DynamicPluginTensorDesc const*, int32_t,
                       DynamicPluginTensorDesc const*,
                       int32_t) noexcept override {}
  size_t getWorkspaceSize(PluginTensorDesc const*, int32_t,
                          PluginTensorDesc const*,
                          int32_t) const noexcept override {
    return 0;
  }
  int32_t enqueue(PluginTensorDesc const* input_desc,
                  PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*,
                  cudaStream_t stream) noexcept override {
    return enqueue_plugin(input_desc, inputs, outputs, stream);
  }
  DataType getOutputDataType(int32_t, DataType const* input_types,
                             int32_t nb_inputs) const noexcept override {
    return nb_inputs == 3 ? input_types[0] : DataType::kFLOAT;
  }
  char const* getPluginType() const noexcept override { return kName; }
  char const* getPluginVersion() const noexcept override { return kVersion; }
  int32_t initialize() noexcept override { return 0; }
  void terminate() noexcept override {}
  size_t getSerializationSize() const noexcept override { return 0; }
  void serialize(void*) const noexcept override {}
  void destroy() noexcept override { delete this; }
  IPluginV2DynamicExt* clone() const noexcept override {
    auto* plugin = new (std::nothrow) DenseScatterPlugin();
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

class DenseScatterCreator final : public IPluginCreator {
 public:
  char const* getPluginName() const noexcept override { return kName; }
  char const* getPluginVersion() const noexcept override { return kVersion; }
  PluginFieldCollection const* getFieldNames() noexcept override {
    static PluginFieldCollection fields{0, nullptr};
    return &fields;
  }
  IPluginV2* createPlugin(char const*,
                          PluginFieldCollection const*) noexcept override {
    auto* plugin = new (std::nothrow) DenseScatterPlugin();
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

using DSVTDenseScatterCreator = dynamic_bevfusion::DenseScatterCreator;
REGISTER_TENSORRT_PLUGIN(DSVTDenseScatterCreator);
