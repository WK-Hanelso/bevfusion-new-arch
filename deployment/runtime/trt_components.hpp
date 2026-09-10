#pragma once

#include <NvInfer.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace thor_multisensor {

class DeviceBuffer;

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, char const* message) noexcept override;
};

template <typename T>
struct TrtDelete {
  void operator()(T* value) const noexcept { delete value; }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

class PluginLibraries {
 public:
  explicit PluginLibraries(std::vector<std::string> const& paths);
  ~PluginLibraries();
  PluginLibraries(PluginLibraries const&) = delete;
  PluginLibraries& operator=(PluginLibraries const&) = delete;

 private:
  std::vector<void*> handles_;
};

struct Engine {
  TrtPtr<nvinfer1::ICudaEngine> engine;
  TrtPtr<nvinfer1::IExecutionContext> context;
};

Engine loadEngine(nvinfer1::IRuntime& runtime, std::string const& path);
void validateAbi(Engine const& camera, Engine const& lidar,
                 Engine const& fusion);
void bind(nvinfer1::IExecutionContext& context, char const* name,
          DeviceBuffer const& buffer);
void enqueue(nvinfer1::IExecutionContext& context, cudaStream_t stream,
             char const* stage);
std::size_t contextDeviceMemoryBytes(Engine const& engine);

}  // namespace thor_multisensor
