#pragma once

#include "cuda_resources.hpp"
#include "options.hpp"
#include "statistics.hpp"
#include "trt_components.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace thor_multisensor {

struct WarmupResult {
  int iterations{0};
  double totalMs{0.0};
};

struct InferenceResult {
  float totalMs{0.0F};
  float cameraStartMs{0.0F};
  float cameraEndMs{0.0F};
  float lidarStartMs{0.0F};
  float lidarEndMs{0.0F};
  float fusionStartMs{0.0F};
  float fusionEndMs{0.0F};
};

struct Detection {
  std::array<float, 9> box{};
  float score{0.0F};
  std::int32_t label{0};
};

struct LatencyReport {
  Summary camera;
  Summary lidar;
  Summary fusion;
  Summary sequential;
  Summary concurrent;
};

class MultisensorPipeline {
 public:
  explicit MultisensorPipeline(Options options);
  ~MultisensorPipeline() = default;
  MultisensorPipeline(MultisensorPipeline const&) = delete;
  MultisensorPipeline& operator=(MultisensorPipeline const&) = delete;

  WarmupResult warmup();
  InferenceResult infer();
  Detection validateOutput();
  LatencyReport benchmark();

  int points() const { return options_.points; }
  bool parallelSubmit() const { return options_.parallelSubmit; }
  std::size_t persistentIoBytes() const;
  std::size_t trtContextDeviceMemoryBytes() const;

 private:
  void submitBranches(Event* cameraBegin, Event* lidarBegin);
  void enqueueConcurrent();
  void checkLidarStatus(char const* stage) const;
  std::vector<float> measureStage(nvinfer1::IExecutionContext& context,
                                  cudaStream_t stream, char const* name);
  std::vector<float> measureSequential();
  std::vector<float> measureConcurrent();

  Options options_;
  // Declared first so plugin libraries are unloaded after TRT objects.
  PluginLibraries plugins_;
  Logger logger_;
  TrtPtr<nvinfer1::IRuntime> runtime_;
  Engine camera_;
  Engine lidar_;
  Engine fusion_;

  DeviceBuffer images_;
  DeviceBuffer geometry_;
  DeviceBuffer points_;
  DeviceBuffer cameraBev_;
  DeviceBuffer lidarBev_;
  DeviceBuffer lidarStatus_;
  DeviceBuffer boxes_;
  DeviceBuffer scores_;
  DeviceBuffer labels_;

  Stream streamA_;
  Stream streamB_;
  Stream streamC_;
  Event cameraReady_;
  Event lidarReady_;
};

}  // namespace thor_multisensor
