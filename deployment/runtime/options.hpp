#pragma once

#include <string>
#include <vector>

namespace thor_multisensor {

struct Options {
  std::string cameraEngine{
      "deployment/artifacts/tensorrt/"
      "camera_bev_fp16.engine"};
  std::string lidarEngine{
      "deployment/artifacts/tensorrt/"
      "dsvt_lidar_fp16.engine"};
  std::string fusionEngine{
      "deployment/artifacts/tensorrt/"
      "fusion_dal_fp16.engine"};
  std::vector<std::string> plugins;
  int points{34688};
  int warmup{20};
  int iterations{100};
  int memorySampleMs{2};
  bool latency{false};
  bool profile{false};
  bool parallelSubmit{true};
  bool syntheticUniquePillars{false};
  bool memory{false};
};

Options parseOptions(int argc, char** argv);
void printUsage(char const* executable);

}  // namespace thor_multisensor
