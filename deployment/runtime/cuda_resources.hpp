#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <string>
#include <thread>

namespace thor_multisensor {

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes);
  ~DeviceBuffer();
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  void* data() const { return data_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void release() noexcept;

  void* data_{nullptr};
  std::size_t bytes_{0};
};

class Stream {
 public:
  Stream();
  ~Stream();
  Stream(Stream const&) = delete;
  Stream& operator=(Stream const&) = delete;
  operator cudaStream_t() const { return value_; }

 private:
  cudaStream_t value_{nullptr};
};

class Event {
 public:
  Event();
  ~Event();
  Event(Event const&) = delete;
  Event& operator=(Event const&) = delete;
  operator cudaEvent_t() const { return value_; }

 private:
  cudaEvent_t value_{nullptr};
};

float elapsed(Event const& begin, Event const& end);

struct MemorySnapshot {
  std::size_t cudaFreeBytes{0};
  std::size_t cudaTotalBytes{0};
  std::size_t systemTotalBytes{0};
  std::size_t systemAvailableBytes{0};
  std::size_t systemBuffersBytes{0};
  std::size_t systemCachedBytes{0};
  std::size_t systemReclaimableSlabBytes{0};
  std::size_t systemSharedBytes{0};
  std::size_t processRssBytes{0};
  std::size_t processHighWaterBytes{0};

  // On integrated Tegra devices this is only cudaMemGetInfo()'s raw
  // total-minus-free view. It is not process usage or physical memory
  // pressure because CPU and iGPU share reclaimable system memory.
  std::size_t cudaNonFreeRawBytes() const {
    return cudaTotalBytes >= cudaFreeBytes ? cudaTotalBytes - cudaFreeBytes : 0;
  }

  std::size_t systemUsedBytes() const {
    return systemTotalBytes >= systemAvailableBytes
               ? systemTotalBytes - systemAvailableBytes
               : 0;
  }

  std::size_t systemReclaimableCacheEstimateBytes() const {
    std::size_t const reclaimable = systemBuffersBytes + systemCachedBytes +
                                    systemReclaimableSlabBytes;
    return reclaimable >= systemSharedBytes ? reclaimable - systemSharedBytes
                                            : 0;
  }
};

MemorySnapshot captureMemorySnapshot();

class MemoryReporter {
 public:
  MemoryReporter(bool enabled, int sampleIntervalMs);
  ~MemoryReporter();
  MemoryReporter(MemoryReporter const&) = delete;
  MemoryReporter& operator=(MemoryReporter const&) = delete;

  bool enabled() const { return enabled_; }
  void printBaseline();
  void beginStage(std::string stage);
  void endStage();
  void printSnapshot(std::string const& stage);
  void printAllocations(std::size_t persistentIoBytes,
                        std::size_t trtContextBytes);

 private:
  void stopNoThrow() noexcept;
  void samplePeakNoThrow() noexcept;
  void print(std::string const& stage, MemorySnapshot const& end,
             MemorySnapshot const* peak) const;

  bool enabled_{false};
  int sampleIntervalMs_{2};
  int device_{0};
  std::string stage_;
  MemorySnapshot baseline_{};
  MemorySnapshot peak_{};
  std::atomic<bool> stop_{true};
  std::thread worker_;
};

}  // namespace thor_multisensor
