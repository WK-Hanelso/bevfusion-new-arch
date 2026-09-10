#include "cuda_resources.hpp"

#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace thor_multisensor {
namespace {

constexpr double kMiB = 1024.0 * 1024.0;

std::size_t readProcStatusBytes(char const* field) {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind(field, 0) != 0) continue;
    std::istringstream values(line.substr(std::char_traits<char>::length(field)));
    std::size_t kib = 0;
    values >> kib;
    return kib * 1024;
  }
  return 0;
}

void readProcMeminfo(MemorySnapshot& result) noexcept {
  std::ifstream meminfo("/proc/meminfo");
  std::string key;
  std::string line;
  std::size_t kib = 0;
  while (std::getline(meminfo, line)) {
    std::istringstream values(line);
    if (!(values >> key >> kib)) continue;
    std::size_t const bytes = kib * 1024;
    if (key == "MemTotal:") {
      result.systemTotalBytes = bytes;
    } else if (key == "MemAvailable:") {
      result.systemAvailableBytes = bytes;
    } else if (key == "Buffers:") {
      result.systemBuffersBytes = bytes;
    } else if (key == "Cached:") {
      result.systemCachedBytes = bytes;
    } else if (key == "SReclaimable:") {
      result.systemReclaimableSlabBytes = bytes;
    } else if (key == "Shmem:") {
      result.systemSharedBytes = bytes;
    }
  }
}

double toMiB(std::size_t bytes) {
  return static_cast<double>(bytes) / kMiB;
}

double systemUsedPercent(MemorySnapshot const& value) {
  if (value.systemTotalBytes == 0) return 0.0;
  return 100.0 * static_cast<double>(value.systemUsedBytes()) /
         static_cast<double>(value.systemTotalBytes);
}

double signedDeltaMiB(std::size_t value, std::size_t baseline) {
  auto const delta = static_cast<long long>(value) -
                     static_cast<long long>(baseline);
  return static_cast<double>(delta) / kMiB;
}

}  // namespace

DeviceBuffer::DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
  require(bytes > 0, "cannot allocate an empty CUDA buffer");
  cudaCheck(cudaMalloc(&data_, bytes), "cudaMalloc");
}

DeviceBuffer::~DeviceBuffer() { release(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    release();
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

void DeviceBuffer::release() noexcept {
  if (data_) cudaFree(data_);
  data_ = nullptr;
  bytes_ = 0;
}

Stream::Stream() {
  cudaCheck(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
}

Stream::~Stream() {
  if (value_) cudaStreamDestroy(value_);
}

Event::Event() { cudaCheck(cudaEventCreate(&value_), "cudaEventCreate"); }

Event::~Event() {
  if (value_) cudaEventDestroy(value_);
}

float elapsed(Event const& begin, Event const& end) {
  float milliseconds = 0.0F;
  cudaCheck(cudaEventElapsedTime(&milliseconds, begin, end),
            "cudaEventElapsedTime");
  return milliseconds;
}

MemorySnapshot captureMemorySnapshot() {
  MemorySnapshot result;
  cudaCheck(cudaMemGetInfo(&result.cudaFreeBytes, &result.cudaTotalBytes),
            "cudaMemGetInfo");
  readProcMeminfo(result);
  result.processRssBytes = readProcStatusBytes("VmRSS:");
  result.processHighWaterBytes = readProcStatusBytes("VmHWM:");
  return result;
}

MemoryReporter::MemoryReporter(bool enabled, int sampleIntervalMs)
    : enabled_(enabled), sampleIntervalMs_(sampleIntervalMs) {
  if (!enabled_) return;
  cudaCheck(cudaGetDevice(&device_), "cudaGetDevice");
  baseline_ = captureMemorySnapshot();
  peak_ = baseline_;
}

MemoryReporter::~MemoryReporter() { stopNoThrow(); }

void MemoryReporter::printBaseline() {
  if (enabled_) print("startup", baseline_, nullptr);
}

void MemoryReporter::beginStage(std::string stage) {
  if (!enabled_) return;
  require(stop_.load(), "memory reporter stage already active");
  stage_ = std::move(stage);
  peak_ = captureMemorySnapshot();
  stop_.store(false);
  worker_ = std::thread([this]() {
    if (cudaSetDevice(device_) != cudaSuccess) return;
    while (!stop_.load()) {
      samplePeakNoThrow();
      std::this_thread::sleep_for(
          std::chrono::milliseconds(sampleIntervalMs_));
    }
    samplePeakNoThrow();
  });
}

void MemoryReporter::endStage() {
  if (!enabled_) return;
  require(!stop_.load(), "memory reporter has no active stage");
  stop_.store(true);
  if (worker_.joinable()) worker_.join();
  MemorySnapshot const end = captureMemorySnapshot();
  peak_.cudaFreeBytes = std::min(peak_.cudaFreeBytes, end.cudaFreeBytes);
  if (end.systemAvailableBytes != 0) {
    if (peak_.systemAvailableBytes == 0) {
      peak_.systemAvailableBytes = end.systemAvailableBytes;
    } else {
      peak_.systemAvailableBytes =
          std::min(peak_.systemAvailableBytes, end.systemAvailableBytes);
    }
  }
  peak_.processRssBytes = std::max(peak_.processRssBytes, end.processRssBytes);
  peak_.processHighWaterBytes =
      std::max(peak_.processHighWaterBytes, end.processHighWaterBytes);
  print(stage_, end, &peak_);
  stage_.clear();
}

void MemoryReporter::printSnapshot(std::string const& stage) {
  if (enabled_) print(stage, captureMemorySnapshot(), nullptr);
}

void MemoryReporter::printAllocations(std::size_t persistentIoBytes,
                                      std::size_t trtContextBytes) {
  if (!enabled_) return;
  std::cout << std::fixed << std::setprecision(2)
            << "memory_allocations persistent_io=" << toMiB(persistentIoBytes)
            << " MiB trt_context_requirement=" << toMiB(trtContextBytes)
            << " MiB tracked_total="
            << toMiB(persistentIoBytes + trtContextBytes) << " MiB\n";
}

void MemoryReporter::stopNoThrow() noexcept {
  stop_.store(true);
  if (worker_.joinable()) worker_.join();
}

void MemoryReporter::samplePeakNoThrow() noexcept {
  MemorySnapshot sample;
  std::size_t freeBytes = 0;
  std::size_t totalBytes = 0;
  if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
    if (peak_.cudaTotalBytes == 0) peak_.cudaTotalBytes = totalBytes;
    peak_.cudaFreeBytes = std::min(peak_.cudaFreeBytes, freeBytes);
  }
  readProcMeminfo(sample);
  if (sample.systemTotalBytes != 0) {
    if (peak_.systemTotalBytes == 0) {
      peak_.systemTotalBytes = sample.systemTotalBytes;
    }
    if (peak_.systemAvailableBytes == 0) {
      peak_.systemAvailableBytes = sample.systemAvailableBytes;
    } else if (sample.systemAvailableBytes != 0) {
      peak_.systemAvailableBytes =
          std::min(peak_.systemAvailableBytes, sample.systemAvailableBytes);
    }
  }
  peak_.processRssBytes =
      std::max(peak_.processRssBytes, readProcStatusBytes("VmRSS:"));
  peak_.processHighWaterBytes =
      std::max(peak_.processHighWaterBytes, readProcStatusBytes("VmHWM:"));
}

void MemoryReporter::print(std::string const& stage,
                           MemorySnapshot const& end,
                           MemorySnapshot const* peak) const {
  std::cout << std::fixed << std::setprecision(2)
            << "memory stage=" << stage
            << " system_used=" << toMiB(end.systemUsedBytes()) << " MiB"
            << " system_available=" << toMiB(end.systemAvailableBytes) << " MiB"
            << " system_total=" << toMiB(end.systemTotalBytes) << " MiB"
            << " system_used_percent=" << systemUsedPercent(end) << "%"
            << " system_reclaimable_cache_estimate="
            << toMiB(end.systemReclaimableCacheEstimateBytes()) << " MiB"
            << " cuda_memgetinfo_free_raw=" << toMiB(end.cudaFreeBytes) << " MiB"
            << " cuda_memgetinfo_nonfree_raw="
            << toMiB(end.cudaNonFreeRawBytes()) << " MiB"
            << " cuda_memgetinfo_total=" << toMiB(end.cudaTotalBytes) << " MiB"
            << " cuda_memgetinfo_nonfree_delta_from_startup="
            << signedDeltaMiB(end.cudaNonFreeRawBytes(),
                              baseline_.cudaNonFreeRawBytes())
            << " MiB"
            << " process_rss=" << toMiB(end.processRssBytes) << " MiB"
            << " process_hwm=" << toMiB(end.processHighWaterBytes) << " MiB";
  if (peak) {
    std::cout << " peak_system_used=" << toMiB(peak->systemUsedBytes())
              << " MiB min_system_available="
              << toMiB(peak->systemAvailableBytes) << " MiB"
              << " peak_cuda_memgetinfo_nonfree_raw="
              << toMiB(peak->cudaNonFreeRawBytes()) << " MiB"
              << " peak_process_rss=" << toMiB(peak->processRssBytes)
              << " MiB";
  }
  std::cout << '\n';
}

}  // namespace thor_multisensor
