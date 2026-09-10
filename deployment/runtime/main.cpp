#include "multisensor_pipeline.hpp"
#include "cuda_resources.hpp"
#include "options.hpp"
#include "statistics.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>

namespace {

using Clock = std::chrono::steady_clock;
using thor_multisensor::MultisensorPipeline;
using thor_multisensor::MemoryReporter;
using thor_multisensor::Options;

void restoreRuntimeNumberFormat() {
  std::cout << std::fixed << std::setprecision(4);
}

void printInference(thor_multisensor::InferenceResult const& value,
                    bool parallelSubmit) {
  std::cout << "inference=" << value.totalMs
            << " ms outputs=boxes[1,200,9],scores[1,200],labels[1,200]\n"
            << "timeline camera_start=" << value.cameraStartMs
            << " camera_end=" << value.cameraEndMs
            << " lidar_start=" << value.lidarStartMs
            << " lidar_end=" << value.lidarEndMs
            << " fusion_start=" << value.fusionStartMs
            << " fusion_end=" << value.fusionEndMs
            << " ms parallel_submit=" << std::boolalpha << parallelSubmit
            << std::noboolalpha << '\n';
}

void printDetection(thor_multisensor::Detection const& value) {
  std::cout << "first_detection score=" << value.score
            << " label=" << value.label << " box=[";
  for (std::size_t i = 0; i < value.box.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << value.box[i];
  }
  std::cout << "]\n";
}

void printLatency(thor_multisensor::LatencyReport const& report,
                  int iterations) {
  thor_multisensor::printSummary("engine_a", report.camera);
  thor_multisensor::printSummary("engine_b", report.lidar);
  thor_multisensor::printSummary("engine_c", report.fusion);
  thor_multisensor::printSummary("sequential", report.sequential);
  thor_multisensor::printSummary("concurrent", report.concurrent);
  std::cout << "median_speedup="
            << report.sequential.p50 / report.concurrent.p50
            << "x iterations=" << iterations << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options const options = thor_multisensor::parseOptions(argc, argv);
    restoreRuntimeNumberFormat();

    MemoryReporter memory(options.memory, options.memorySampleMs);
    memory.printBaseline();

    auto const initializationBegin = Clock::now();
    memory.beginStage("initialization");
    auto pipeline = std::make_unique<MultisensorPipeline>(options);
    memory.endStage();
    restoreRuntimeNumberFormat();
    auto const initializationEnd = Clock::now();
    double const initializationMs =
        std::chrono::duration<double, std::milli>(initializationEnd -
                                                  initializationBegin)
            .count();
    std::cout << "initialization=" << initializationMs << " ms"
              << " points=" << pipeline->points() << '\n';
    memory.printAllocations(pipeline->persistentIoBytes(),
                            pipeline->trtContextDeviceMemoryBytes());

    memory.beginStage("warmup");
    auto const warmup = pipeline->warmup();
    memory.endStage();
    restoreRuntimeNumberFormat();
    std::cout << "warmup iterations=" << warmup.iterations
              << " total=" << warmup.totalMs << " ms";
    if (warmup.iterations > 0) {
      std::cout << " average=" << warmup.totalMs / warmup.iterations << " ms";
    }
    std::cout << '\n';

    memory.beginStage("inference");
    auto const inference = pipeline->infer();
    memory.endStage();
    restoreRuntimeNumberFormat();
    printInference(inference, pipeline->parallelSubmit());
    printDetection(pipeline->validateOutput());

    if (options.latency) printLatency(pipeline->benchmark(), options.iterations);

    pipeline.reset();
    memory.printSnapshot("shutdown");
    std::cout << "PASS C++ TensorRT inference\n";
    return EXIT_SUCCESS;
  } catch (std::exception const& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
