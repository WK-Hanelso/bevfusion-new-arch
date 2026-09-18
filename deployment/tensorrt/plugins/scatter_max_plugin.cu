/* TensorRT V2DynamicExt/V3 scatter-max used by both DynamicPFN layers. */

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <new>
#include <string>

namespace dynamic_bevfusion {
namespace {
using namespace nvinfer1;

constexpr char kName[] = "DynamicScatterMax";
constexpr char kVersion[] = "1";
#ifndef DYNAMIC_BEVFUSION_MAX_PILLARS
#define DYNAMIC_BEVFUSION_MAX_PILLARS 10000
#endif
constexpr int kMaxPillars = DYNAMIC_BEVFUSION_MAX_PILLARS;

__device__ float atomic_max_float(float* address, float value) {
  int* integer_address = reinterpret_cast<int*>(address);
  int old = *integer_address;
  while (__int_as_float(old) < value) {
    int assumed = old;
    old = atomicCAS(integer_address, assumed, __float_as_int(value));
    if (old == assumed) break;
  }
  return __int_as_float(old);
}

__global__ void initialize_negative_infinity(float* output, int values) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < values) output[index] = -__int_as_float(0x7f800000);
}

__global__ void scatter_max_kernel(const float* features, const int* inverse,
                                   int points, int channels, float* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int values = points * channels;
  if (index >= values) return;
  int point = index / channels;
  int pillar = inverse[point];
  if (pillar < 0 || pillar >= kMaxPillars) return;
  atomic_max_float(output + pillar * channels + index % channels, features[index]);
}

int32_t enqueue_plugin(PluginTensorDesc const* input_desc,
                       void const* const* inputs, void* const* outputs,
                       cudaStream_t stream) {
  if (input_desc[0].dims.nbDims != 2 || input_desc[1].dims.nbDims != 1 ||
      input_desc[0].dims.d[0] != input_desc[1].dims.d[0]) {
    return 1;
  }
  int points = input_desc[0].dims.d[0];
  int channels = input_desc[0].dims.d[1];
  int output_values = kMaxPillars * channels;
  initialize_negative_infinity<<<(output_values + 255) / 256, 256, 0,
                                 stream>>>(static_cast<float*>(outputs[0]),
                                           output_values);
  int values = points * channels;
  scatter_max_kernel<<<(values + 255) / 256, 256, 0, stream>>>(
      static_cast<float const*>(inputs[0]), static_cast<int const*>(inputs[1]),
      points, channels, static_cast<float*>(outputs[0]));
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

#if NV_TENSORRT_MAJOR >= 10

class ScatterMaxPlugin final : public IPluginV3,
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
  IPluginV3* clone() noexcept override { return new (std::nothrow) ScatterMaxPlugin(); }
  AsciiChar const* getPluginName() const noexcept override { return kName; }
  AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t,
                          DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }
  int32_t getOutputDataTypes(DataType* output_types, int32_t outputs,
                             DataType const*, int32_t) const noexcept override {
    if (outputs != 1) return 1;
    output_types[0] = DataType::kFLOAT;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* inputs, int32_t nb_inputs,
                          DimsExprs const*, int32_t, DimsExprs* outputs,
                          int32_t nb_outputs, IExprBuilder& builder) noexcept override {
    if (nb_inputs != 2 || nb_outputs != 1 || inputs[0].nbDims != 2) return 1;
    outputs[0].nbDims = 2;
    outputs[0].d[0] = builder.constant(kMaxPillars);
    outputs[0].d[1] = inputs[0].d[1];
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io,
                                 int32_t inputs, int32_t outputs) noexcept override {
    if (inputs != 2 || outputs != 1 || pos < 0 || pos >= 3) return false;
    DataType expected = pos == 1 ? DataType::kINT32 : DataType::kFLOAT;
    return io[pos].desc.format == TensorFormat::kLINEAR && io[pos].desc.type == expected;
  }
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t onShapeChange(PluginTensorDesc const* in, int32_t inputs,
                        PluginTensorDesc const*, int32_t outputs) noexcept override {
    return (inputs == 2 && outputs == 1 && in[0].dims.nbDims == 2 &&
            in[1].dims.nbDims == 1 && in[0].dims.d[0] == in[1].dims.d[0]) ? 0 : 1;
  }
  int32_t enqueue(PluginTensorDesc const* input_desc, PluginTensorDesc const*,
                  void const* const* inputs, void* const* outputs, void*,
                  cudaStream_t stream) noexcept override {
    return enqueue_plugin(input_desc, inputs, outputs, stream);
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override {
    static PluginFieldCollection fields{0, nullptr};
    return &fields;
  }
};

class ScatterMaxCreator final : public IPluginCreatorV3One {
 public:
  IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const*,
                          TensorRTPhase) noexcept override {
    return new (std::nothrow) ScatterMaxPlugin();
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

class ScatterMaxPlugin final : public IPluginV2DynamicExt {
 public:
  int32_t getNbOutputs() const noexcept override { return 1; }
  DimsExprs getOutputDimensions(int32_t output_index,
                                DimsExprs const* inputs, int32_t nb_inputs,
                                IExprBuilder& builder) noexcept override {
    DimsExprs output{};
    if (output_index != 0 || nb_inputs != 2 || inputs[0].nbDims != 2) {
      return output;
    }
    output.nbDims = 2;
    output.d[0] = builder.constant(kMaxPillars);
    output.d[1] = inputs[0].d[1];
    return output;
  }
  bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* io,
                                 int32_t inputs,
                                 int32_t outputs) noexcept override {
    if (inputs != 2 || outputs != 1 || pos < 0 || pos >= 3) return false;
    DataType expected = pos == 1 ? DataType::kINT32 : DataType::kFLOAT;
    return io[pos].format == TensorFormat::kLINEAR && io[pos].type == expected;
  }
  void configurePlugin(DynamicPluginTensorDesc const*, int32_t,
                       DynamicPluginTensorDesc const*,
                       int32_t) noexcept override {}
  size_t getWorkspaceSize(PluginTensorDesc const*, int32_t,
                          PluginTensorDesc const*,
                          int32_t) const noexcept override {
    return 0;
  }
  int32_t enqueue(PluginTensorDesc const* input_desc, PluginTensorDesc const*,
                  void const* const* inputs, void* const* outputs, void*,
                  cudaStream_t stream) noexcept override {
    return enqueue_plugin(input_desc, inputs, outputs, stream);
  }
  DataType getOutputDataType(int32_t, DataType const*,
                             int32_t) const noexcept override {
    return DataType::kFLOAT;
  }
  char const* getPluginType() const noexcept override { return kName; }
  char const* getPluginVersion() const noexcept override { return kVersion; }
  int32_t initialize() noexcept override { return 0; }
  void terminate() noexcept override {}
  size_t getSerializationSize() const noexcept override { return 0; }
  void serialize(void*) const noexcept override {}
  void destroy() noexcept override { delete this; }
  IPluginV2DynamicExt* clone() const noexcept override {
    auto* plugin = new (std::nothrow) ScatterMaxPlugin();
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

class ScatterMaxCreator final : public IPluginCreator {
 public:
  char const* getPluginName() const noexcept override { return kName; }
  char const* getPluginVersion() const noexcept override { return kVersion; }
  PluginFieldCollection const* getFieldNames() noexcept override {
    static PluginFieldCollection fields{0, nullptr};
    return &fields;
  }
  IPluginV2* createPlugin(char const*,
                          PluginFieldCollection const*) noexcept override {
    auto* plugin = new (std::nothrow) ScatterMaxPlugin();
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

using DynamicScatterMaxCreator = dynamic_bevfusion::ScatterMaxCreator;
REGISTER_TENSORRT_PLUGIN(DynamicScatterMaxCreator);
