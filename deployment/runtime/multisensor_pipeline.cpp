#include "multisensor_pipeline.hpp"

#include "common.hpp"
#include "point_features.hpp"

#include <NvInferPlugin.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <string>
#include <thread>
#include <utility>

namespace thor_multisensor {
namespace {

using Clock = std::chrono::steady_clock;

std::vector<float> makeSyntheticPoints(int count, bool uniquePillars) {
  std::vector<float> points(static_cast<std::size_t>(count) * 5);
  // Deterministic 80x80 XY grid (at most 6,400 occupied pillars), repeated as
  // needed. It exercises the dynamic point profile without requiring a dataset.
  for (int i = 0; i < count; ++i) {
    int const cell = uniquePillars ? i % (360 * 360) : i % 6400;
    if (uniquePillars) {
      points[static_cast<std::size_t>(i) * 5 + 0] =
          -54.0F + (static_cast<float>(cell / 360) + 0.25F) * 0.3F;
      points[static_cast<std::size_t>(i) * 5 + 1] =
          -54.0F + (static_cast<float>(cell % 360) + 0.25F) * 0.3F;
    } else {
      points[static_cast<std::size_t>(i) * 5 + 0] =
          -47.4F + 1.2F * static_cast<float>(cell % 80);
      points[static_cast<std::size_t>(i) * 5 + 1] =
          -47.4F + 1.2F * static_cast<float>(cell / 80);
    }
    points[static_cast<std::size_t>(i) * 5 + 2] =
        -2.0F + 0.5F * static_cast<float>(i % 8);
    points[static_cast<std::size_t>(i) * 5 + 3] =
        static_cast<float>(i % 256) / 255.0F;
    points[static_cast<std::size_t>(i) * 5 + 4] = 0.0F;
  }
  return points;
}

}  // namespace

MultisensorPipeline::MultisensorPipeline(Options options)
    : options_(std::move(options)), plugins_(options_.plugins) {
  require(initLibNvInferPlugins(&logger_, ""),
          "failed to initialize TensorRT standard plugins");
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  require(runtime_ != nullptr, "failed to create TensorRT runtime");

  camera_ = loadEngine(*runtime_, options_.cameraEngine);
  lidar_ = loadEngine(*runtime_, options_.lidarEngine);
  fusion_ = loadEngine(*runtime_, options_.fusionEngine);
  validateAbi(camera_, lidar_, fusion_);
  require(lidar_.context->setInputShape(
              "points", nvinfer1::Dims2{options_.points, 5}),
          "point count is outside the Engine B optimization profile");

  images_ = DeviceBuffer(1ULL * 6 * 3 * 256 * 704 * sizeof(float));
  geometry_ = DeviceBuffer(1ULL * 6 * 59 * 16 * 44 * 3 * sizeof(float));
  points_ = DeviceBuffer(static_cast<std::size_t>(options_.points) * 5 *
                         sizeof(float));
  cameraBev_ = DeviceBuffer(1ULL * 80 * 180 * 180 * sizeof(float));
  lidarBev_ = DeviceBuffer(1ULL * 256 * 180 * 180 * sizeof(float));
  lidarStatus_ = DeviceBuffer(sizeof(std::int32_t));
  boxes_ = DeviceBuffer(1ULL * 200 * 9 * sizeof(float));
  scores_ = DeviceBuffer(1ULL * 200 * sizeof(float));
  labels_ = DeviceBuffer(1ULL * 200 * sizeof(std::int32_t));

  cudaCheck(cudaMemset(images_.data(), 0, images_.bytes()), "zero images");
  cudaCheck(cudaMemset(geometry_.data(), 0, geometry_.bytes()),
            "zero geometry");
  cudaCheck(cudaMemset(lidarStatus_.data(), 0xff, lidarStatus_.bytes()),
            "initialize lidar status");
  auto hostPoints = makeSyntheticPoints(options_.points,
                                        options_.syntheticUniquePillars);
  auto zeroFeatureChannels = options_.zeroFeatureChannels;
  if (!options_.lidarManifest.empty()) {
    require(zeroFeatureChannels.empty(),
            "manifest and direct zero-feature options are mutually exclusive");
    zeroFeatureChannels = zeroFeatureChannelsFromManifest(
        options_.lidarManifest, 5);
  }
  zeroPointFeatureChannels(hostPoints, 5, zeroFeatureChannels);
  cudaCheck(cudaMemcpy(points_.data(), hostPoints.data(), points_.bytes(),
                       cudaMemcpyHostToDevice),
            "upload synthetic points");

  bind(*camera_.context, "images", images_);
  bind(*camera_.context, "geometry", geometry_);
  bind(*camera_.context, "camera_bev", cameraBev_);
  bind(*lidar_.context, "points", points_);
  bind(*lidar_.context, "lidar_bev", lidarBev_);
  bind(*lidar_.context, "lidar_status", lidarStatus_);
  // Engine C reads A/B output allocations directly. No intermediate copy.
  bind(*fusion_.context, "camera_bev", cameraBev_);
  bind(*fusion_.context, "lidar_bev", lidarBev_);
  bind(*fusion_.context, "boxes", boxes_);
  bind(*fusion_.context, "scores", scores_);
  bind(*fusion_.context, "labels", labels_);
}

WarmupResult MultisensorPipeline::warmup() {
  auto const begin = Clock::now();
  for (int i = 0; i < options_.warmup; ++i) {
    enqueueConcurrent();
    cudaCheck(cudaStreamSynchronize(streamC_), "warmup synchronize");
  }
  if (options_.warmup > 0) checkLidarStatus("warmup");
  auto const end = Clock::now();
  return {options_.warmup,
          std::chrono::duration<double, std::milli>(end - begin).count()};
}

InferenceResult MultisensorPipeline::infer() {
  Event inferenceBegin;
  Event inferenceEnd;
  Event cameraBegin;
  Event lidarBegin;
  Event fusionBegin;
  if (options_.profile) cudaCheck(cudaProfilerStart(), "cudaProfilerStart");
  cudaCheck(cudaEventRecord(inferenceBegin, streamC_),
            "record inference start");
  cudaCheck(cudaStreamWaitEvent(streamA_, inferenceBegin),
            "gate camera stream");
  cudaCheck(cudaStreamWaitEvent(streamB_, inferenceBegin), "gate lidar stream");
  submitBranches(&cameraBegin, &lidarBegin);
  cudaCheck(cudaStreamWaitEvent(streamC_, cameraReady_), "wait camera event");
  cudaCheck(cudaStreamWaitEvent(streamC_, lidarReady_), "wait lidar event");
  cudaCheck(cudaEventRecord(fusionBegin, streamC_), "record fusion start");
  enqueue(*fusion_.context, streamC_, "Engine C");
  cudaCheck(cudaEventRecord(inferenceEnd, streamC_), "record inference end");
  cudaCheck(cudaEventSynchronize(inferenceEnd), "inference synchronize");
  if (options_.profile) cudaCheck(cudaProfilerStop(), "cudaProfilerStop");
  checkLidarStatus("inference");

  return {elapsed(inferenceBegin, inferenceEnd),
          elapsed(inferenceBegin, cameraBegin),
          elapsed(inferenceBegin, cameraReady_),
          elapsed(inferenceBegin, lidarBegin),
          elapsed(inferenceBegin, lidarReady_),
          elapsed(inferenceBegin, fusionBegin),
          elapsed(inferenceBegin, inferenceEnd)};
}

Detection MultisensorPipeline::validateOutput() {
  std::vector<float> hostBoxes(200 * 9);
  std::vector<float> hostScores(200);
  std::vector<std::int32_t> hostLabels(200);
  cudaCheck(cudaMemcpy(hostBoxes.data(), boxes_.data(), boxes_.bytes(),
                       cudaMemcpyDeviceToHost),
            "download boxes");
  cudaCheck(cudaMemcpy(hostScores.data(), scores_.data(), scores_.bytes(),
                       cudaMemcpyDeviceToHost),
            "download scores");
  cudaCheck(cudaMemcpy(hostLabels.data(), labels_.data(), labels_.bytes(),
                       cudaMemcpyDeviceToHost),
            "download labels");
  require(std::all_of(hostBoxes.begin(), hostBoxes.end(),
                      [](float value) { return std::isfinite(value); }),
          "non-finite box output");
  require(std::all_of(hostScores.begin(), hostScores.end(), [](float value) {
            return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
          }),
          "invalid score output");
  require(std::all_of(hostLabels.begin(), hostLabels.end(),
                      [](std::int32_t value) {
                        return value >= 0 && value < 10;
                      }),
          "invalid class label output");

  Detection result;
  std::copy_n(hostBoxes.begin(), result.box.size(), result.box.begin());
  result.score = hostScores.front();
  result.label = hostLabels.front();
  return result;
}

LatencyReport MultisensorPipeline::benchmark() {
  LatencyReport report{
      summarize(measureStage(*camera_.context, streamA_, "Engine A")),
      summarize(measureStage(*lidar_.context, streamB_, "Engine B")),
      summarize(measureStage(*fusion_.context, streamC_, "Engine C")),
      summarize(measureSequential()), summarize(measureConcurrent())};
  checkLidarStatus("benchmark");
  return report;
}

std::size_t MultisensorPipeline::persistentIoBytes() const {
  return images_.bytes() + geometry_.bytes() + points_.bytes() +
         cameraBev_.bytes() + lidarBev_.bytes() + lidarStatus_.bytes() +
         boxes_.bytes() + scores_.bytes() + labels_.bytes();
}

void MultisensorPipeline::checkLidarStatus(char const* stage) const {
  std::int32_t status = -1;
  cudaCheck(cudaMemcpy(&status, lidarStatus_.data(), sizeof(status),
                       cudaMemcpyDeviceToHost),
            "download lidar status");
  require(status == 0,
          std::string(stage) + " rejected: lidar_status=" +
              std::to_string(status) + " (LiDAR capacity overflow)");
}

std::size_t MultisensorPipeline::trtContextDeviceMemoryBytes() const {
  return contextDeviceMemoryBytes(camera_) + contextDeviceMemoryBytes(lidar_) +
         contextDeviceMemoryBytes(fusion_);
}

void MultisensorPipeline::submitBranches(Event* cameraBegin, Event* lidarBegin) {
  auto cameraWork = [&]() {
    if (cameraBegin) {
      cudaCheck(cudaEventRecord(*cameraBegin, streamA_), "record camera start");
    }
    enqueue(*camera_.context, streamA_, "Engine A");
    cudaCheck(cudaEventRecord(cameraReady_, streamA_), "record camera event");
  };
  auto lidarWork = [&]() {
    if (lidarBegin) {
      cudaCheck(cudaEventRecord(*lidarBegin, streamB_), "record lidar start");
    }
    enqueue(*lidar_.context, streamB_, "Engine B");
    cudaCheck(cudaEventRecord(lidarReady_, streamB_), "record lidar event");
  };
  if (!options_.parallelSubmit) {
    cameraWork();
    lidarWork();
    return;
  }

  std::exception_ptr cameraError;
  std::exception_ptr lidarError;
  std::thread cameraThread([&]() {
    try {
      cameraWork();
    } catch (...) {
      cameraError = std::current_exception();
    }
  });
  std::thread lidarThread([&]() {
    try {
      lidarWork();
    } catch (...) {
      lidarError = std::current_exception();
    }
  });
  cameraThread.join();
  lidarThread.join();
  if (cameraError) std::rethrow_exception(cameraError);
  if (lidarError) std::rethrow_exception(lidarError);
}

void MultisensorPipeline::enqueueConcurrent() {
  submitBranches(nullptr, nullptr);
  cudaCheck(cudaStreamWaitEvent(streamC_, cameraReady_), "wait camera event");
  cudaCheck(cudaStreamWaitEvent(streamC_, lidarReady_), "wait lidar event");
  enqueue(*fusion_.context, streamC_, "Engine C");
}

std::vector<float> MultisensorPipeline::measureStage(
    nvinfer1::IExecutionContext& context, cudaStream_t stream,
    char const* name) {
  Event begin;
  Event end;
  std::vector<float> samples;
  samples.reserve(static_cast<std::size_t>(options_.iterations));
  for (int i = 0; i < options_.iterations; ++i) {
    cudaCheck(cudaEventRecord(begin, stream), "record stage start");
    enqueue(context, stream, name);
    cudaCheck(cudaEventRecord(end, stream), "record stage end");
    cudaCheck(cudaEventSynchronize(end), "stage synchronize");
    samples.push_back(elapsed(begin, end));
  }
  return samples;
}

std::vector<float> MultisensorPipeline::measureSequential() {
  Event begin;
  Event end;
  std::vector<float> samples;
  samples.reserve(static_cast<std::size_t>(options_.iterations));
  for (int i = 0; i < options_.iterations; ++i) {
    cudaCheck(cudaEventRecord(begin, streamC_), "record sequential start");
    enqueue(*camera_.context, streamC_, "Engine A sequential");
    enqueue(*lidar_.context, streamC_, "Engine B sequential");
    enqueue(*fusion_.context, streamC_, "Engine C sequential");
    cudaCheck(cudaEventRecord(end, streamC_), "record sequential end");
    cudaCheck(cudaEventSynchronize(end), "sequential synchronize");
    samples.push_back(elapsed(begin, end));
  }
  return samples;
}

std::vector<float> MultisensorPipeline::measureConcurrent() {
  Event begin;
  Event end;
  std::vector<float> samples;
  samples.reserve(static_cast<std::size_t>(options_.iterations));
  for (int i = 0; i < options_.iterations; ++i) {
    cudaCheck(cudaEventRecord(begin, streamC_), "record concurrent start");
    cudaCheck(cudaStreamWaitEvent(streamA_, begin), "gate camera stream");
    cudaCheck(cudaStreamWaitEvent(streamB_, begin), "gate lidar stream");
    enqueueConcurrent();
    cudaCheck(cudaEventRecord(end, streamC_), "record concurrent end");
    cudaCheck(cudaEventSynchronize(end), "concurrent synchronize");
    samples.push_back(elapsed(begin, end));
  }
  return samples;
}

}  // namespace thor_multisensor
