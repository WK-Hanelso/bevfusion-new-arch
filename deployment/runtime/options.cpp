#include "options.hpp"

#include "common.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace thor_multisensor {
namespace {

int parsePositive(std::string const& value, char const* option,
                  bool allowZero) {
  std::size_t consumed = 0;
  int parsed = 0;
  try {
    parsed = std::stoi(value, &consumed);
  } catch (...) {
    throw std::runtime_error(std::string("invalid value for ") + option);
  }
  require(consumed == value.size() && (allowZero ? parsed >= 0 : parsed > 0),
          std::string("invalid value for ") + option + ": " + value);
  return parsed;
}

}  // namespace

void printUsage(char const* executable) {
  std::cout
      << "Usage: " << executable << " [options]\n\n"
      << "Options:\n"
      << "  --camera-engine PATH   Engine A path\n"
      << "  --lidar-engine PATH    Engine B path\n"
      << "  --fusion-engine PATH   Engine C path\n"
      << "  --lidar-manifest PATH  Verified Engine B build manifest\n"
      << "  --plugin PATH          Engine B plugin; repeat five times\n"
      << "  --points N             Synthetic LiDAR point count (default 34688)\n"
      << "  --zero-feature-channel N  Zero this point feature before H2D; repeatable\n"
      << "  --synthetic-unique-pillars  Maximize occupied synthetic pillars\n"
      << "  --warmup N             Concurrent warmup iterations (default 20)\n"
      << "  --latency              Measure A/B/C, sequential, concurrent latency\n"
      << "  --iterations N         Measured iterations (default 100)\n"
      << "  --memory               Report CUDA/RSS memory and stage peaks\n"
      << "  --memory-sample-ms N   Peak polling period; enables --memory (default 2)\n"
      << "  --profile              Delimit one inference with cudaProfilerStart/Stop\n"
      << "  --parallel-submit      Submit A/B from host threads (default)\n"
      << "  --single-thread-submit Submit A/B sequentially from one host thread\n"
      << "  --help                  Show this help\n\n"
      << "Without --plugin, deployment/artifacts/tensorrt/plugins libraries are used.\n";
}

Options parseOptions(int argc, char** argv) {
  Options options;
  auto valueAfter = [&](int& index, char const* option) -> std::string {
    require(index + 1 < argc, std::string("missing value for ") + option);
    return argv[++index];
  };
  for (int i = 1; i < argc; ++i) {
    std::string argument = argv[i];
    if (argument == "--camera-engine") {
      options.cameraEngine = valueAfter(i, argument.c_str());
    } else if (argument == "--lidar-engine") {
      options.lidarEngine = valueAfter(i, argument.c_str());
    } else if (argument == "--fusion-engine") {
      options.fusionEngine = valueAfter(i, argument.c_str());
    } else if (argument == "--lidar-manifest") {
      options.lidarManifest = valueAfter(i, argument.c_str());
    } else if (argument == "--plugin") {
      options.plugins.push_back(valueAfter(i, argument.c_str()));
    } else if (argument == "--points") {
      options.points = parsePositive(valueAfter(i, argument.c_str()),
                                     argument.c_str(), false);
    } else if (argument == "--zero-feature-channel") {
      int const channel = parsePositive(valueAfter(i, argument.c_str()),
                                        argument.c_str(), true);
      require(channel < 5, "--zero-feature-channel must be in [0,4]");
      require(std::find(options.zeroFeatureChannels.begin(),
                        options.zeroFeatureChannels.end(), channel) ==
                  options.zeroFeatureChannels.end(),
              "duplicate --zero-feature-channel");
      options.zeroFeatureChannels.push_back(channel);
    } else if (argument == "--warmup") {
      options.warmup = parsePositive(valueAfter(i, argument.c_str()),
                                     argument.c_str(), true);
    } else if (argument == "--iterations") {
      options.iterations = parsePositive(valueAfter(i, argument.c_str()),
                                         argument.c_str(), false);
    } else if (argument == "--memory-sample-ms") {
      options.memorySampleMs = parsePositive(valueAfter(i, argument.c_str()),
                                             argument.c_str(), false);
      options.memory = true;
    } else if (argument == "--latency") {
      options.latency = true;
    } else if (argument == "--memory") {
      options.memory = true;
    } else if (argument == "--synthetic-unique-pillars") {
      options.syntheticUniquePillars = true;
    } else if (argument == "--profile") {
      options.profile = true;
    } else if (argument == "--parallel-submit") {
      options.parallelSubmit = true;
    } else if (argument == "--single-thread-submit") {
      options.parallelSubmit = false;
    } else if (argument == "--help" || argument == "-h") {
      printUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (options.plugins.empty()) {
    std::string const root = "deployment/artifacts/tensorrt/plugins/";
    options.plugins = {
        root + "libdynamic_pillar_decorate.so",
        root + "libdynamic_scatter_max.so",
        root + "libdsvt_rotated_set.so",
        root + "libtensor_barrier.so",
        root + "libdsvt_dense_scatter.so",
    };
  }
  require(options.plugins.size() == 5,
          "exactly five --plugin paths are required for Engine B");
  return options;
}

}  // namespace thor_multisensor
