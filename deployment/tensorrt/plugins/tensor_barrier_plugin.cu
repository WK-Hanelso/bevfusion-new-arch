/*
 * TensorRT 10 IPluginV3 identity layer used as an internal optimizer barrier.
 *
 * This plugin exists to prevent a TensorRT 10.13/Myelin fusion bug in the
 * DynamicPillarVFE graph without exposing a diagnostic tensor as an engine
 * output. It preserves shape and dtype and performs one asynchronous D2D copy.
 */

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <new>

namespace dynamic_bevfusion {
namespace {

using namespace nvinfer1;

constexpr char kName[] = "TensorBarrier";
constexpr char kVersion[] = "1";

__global__ void copy_bytes(const std::uint8_t* input, std::uint8_t* output,
                           std::size_t count) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (index < count) output[index] = input[index];
}

std::size_t element_size(DataType type) {
  switch (type) {
    case DataType::kFLOAT:
      return 4;
    case DataType::kHALF:
      return 2;
    default:
      return 0;
  }
}

class TensorBarrierPlugin final : public IPluginV3,
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
    return new (std::nothrow) TensorBarrierPlugin();
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
    if (nb_inputs != 1 || nb_outputs != 1) return 1;
    output_types[0] = input_types[0];
    return 0;
  }

  int32_t getOutputShapes(DimsExprs const* inputs, int32_t nb_inputs,
                          DimsExprs const*, int32_t, DimsExprs* outputs,
                          int32_t nb_outputs,
                          IExprBuilder&) noexcept override {
    if (nb_inputs != 1 || nb_outputs != 1) return 1;
    outputs[0] = inputs[0];
    return 0;
  }

  bool supportsFormatCombination(int32_t pos,
                                 DynamicPluginTensorDesc const* io,
                                 int32_t nb_inputs,
                                 int32_t nb_outputs) noexcept override {
    if (nb_inputs != 1 || nb_outputs != 1 || pos < 0 || pos >= 2) return false;
    DataType type = io[pos].desc.type;
    bool supported = type == DataType::kFLOAT || type == DataType::kHALF;
    return supported && io[pos].desc.format == TensorFormat::kLINEAR &&
           (pos == 0 || type == io[0].desc.type);
  }

  int32_t getNbOutputs() const noexcept override { return 1; }

  int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nb_inputs,
                        PluginTensorDesc const* outputs,
                        int32_t nb_outputs) noexcept override {
    if (nb_inputs != 1 || nb_outputs != 1 ||
        inputs[0].dims.nbDims != outputs[0].dims.nbDims ||
        inputs[0].type != outputs[0].type ||
        element_size(inputs[0].type) == 0) {
      return 1;
    }
    for (int index = 0; index < inputs[0].dims.nbDims; ++index) {
      if (inputs[0].dims.d[index] < 0 ||
          inputs[0].dims.d[index] != outputs[0].dims.d[index]) {
        return 1;
      }
    }
    return 0;
  }

  int32_t enqueue(PluginTensorDesc const* input_desc,
                  PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*,
                  cudaStream_t stream) noexcept override {
    std::size_t elements = 1;
    for (int index = 0; index < input_desc[0].dims.nbDims; ++index) {
      int extent = input_desc[0].dims.d[index];
      if (extent < 0) return 1;
      elements *= static_cast<std::size_t>(extent);
    }
    std::size_t bytes = elements * element_size(input_desc[0].type);
    if (bytes == 0) return 0;
    constexpr int threads = 256;
    int blocks = static_cast<int>((bytes + threads - 1) / threads);
    copy_bytes<<<blocks, threads, 0, stream>>>(
        static_cast<const std::uint8_t*>(inputs[0]),
        static_cast<std::uint8_t*>(outputs[0]), bytes);
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

class TensorBarrierCreator final : public IPluginCreatorV3One {
 public:
  IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const*,
                          TensorRTPhase) noexcept override {
    return new (std::nothrow) TensorBarrierPlugin();
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

}  // namespace
}  // namespace dynamic_bevfusion

using TensorBarrierCreator = dynamic_bevfusion::TensorBarrierCreator;
REGISTER_TENSORRT_PLUGIN(TensorBarrierCreator);
