/* TensorRT 10 IPluginV3 scatter-max used by both DynamicPFN layers. */

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <new>

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
    int points = input_desc[0].dims.d[0];
    int channels = input_desc[0].dims.d[1];
    int output_values = kMaxPillars * channels;
    initialize_negative_infinity<<<(output_values + 255) / 256, 256, 0, stream>>>(
        static_cast<float*>(outputs[0]), output_values);
    int values = points * channels;
    scatter_max_kernel<<<(values + 255) / 256, 256, 0, stream>>>(
        static_cast<float const*>(inputs[0]), static_cast<int const*>(inputs[1]),
        points, channels, static_cast<float*>(outputs[0]));
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
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
}  // namespace
}  // namespace dynamic_bevfusion

using DynamicScatterMaxCreator = dynamic_bevfusion::ScatterMaxCreator;
REGISTER_TENSORRT_PLUGIN(DynamicScatterMaxCreator);
