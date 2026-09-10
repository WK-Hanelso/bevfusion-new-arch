#include "trt_components.hpp"

#include "common.hpp"
#include "cuda_resources.hpp"

#include <dlfcn.h>

#include <fstream>
#include <iostream>
#include <utility>

namespace thor_multisensor {
namespace {

std::vector<char> readFile(std::string const& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  require(stream.good(), "failed to open " + path);
  auto const end = stream.tellg();
  require(end > 0, "empty engine file: " + path);
  std::vector<char> data(static_cast<std::size_t>(end));
  stream.seekg(0, std::ios::beg);
  stream.read(data.data(), static_cast<std::streamsize>(data.size()));
  require(stream.good(), "failed to read " + path);
  return data;
}

std::string dimsString(nvinfer1::Dims const& dimensions) {
  std::string result = "[";
  for (int i = 0; i < dimensions.nbDims; ++i) {
    if (i) result += ',';
    result += std::to_string(dimensions.d[i]);
  }
  return result + ']';
}

void expectTensor(nvinfer1::ICudaEngine const& engine, char const* name,
                  nvinfer1::TensorIOMode mode, nvinfer1::DataType type,
                  std::vector<int> const& expected) {
  require(engine.getTensorIOMode(name) == mode,
          std::string("missing or wrong I/O mode for tensor ") + name);
  require(engine.getTensorDataType(name) == type,
          std::string("wrong dtype for tensor ") + name);
  auto dimensions = engine.getTensorShape(name);
  require(dimensions.nbDims == static_cast<int>(expected.size()),
          std::string("wrong rank for tensor ") + name);
  for (int i = 0; i < dimensions.nbDims; ++i) {
    require(dimensions.d[i] == expected[static_cast<std::size_t>(i)],
            std::string("wrong shape for tensor ") + name + ": " +
                dimsString(dimensions));
  }
}

}  // namespace

void Logger::log(Severity severity, char const* message) noexcept {
  if (severity <= Severity::kWARNING) {
    std::cerr << "[TensorRT] " << message << '\n';
  }
}

PluginLibraries::PluginLibraries(std::vector<std::string> const& paths) {
  handles_.reserve(paths.size());
  try {
    for (auto const& path : paths) {
      dlerror();
      void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
      char const* error = dlerror();
      require(handle != nullptr && error == nullptr,
              "failed to load plugin " + path +
                  (error ? std::string(": ") + error : std::string()));
      handles_.push_back(handle);
    }
  } catch (...) {
    for (auto iterator = handles_.rbegin(); iterator != handles_.rend();
         ++iterator) {
      dlclose(*iterator);
    }
    throw;
  }
}

PluginLibraries::~PluginLibraries() {
  for (auto iterator = handles_.rbegin(); iterator != handles_.rend();
       ++iterator) {
    dlclose(*iterator);
  }
}

Engine loadEngine(nvinfer1::IRuntime& runtime, std::string const& path) {
  auto bytes = readFile(path);
  TrtPtr<nvinfer1::ICudaEngine> engine(
      runtime.deserializeCudaEngine(bytes.data(), bytes.size()));
  require(engine != nullptr, "failed to deserialize " + path);
  TrtPtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
  require(context != nullptr, "failed to create execution context for " + path);
  return {std::move(engine), std::move(context)};
}

void validateAbi(Engine const& camera, Engine const& lidar,
                 Engine const& fusion) {
  using Type = nvinfer1::DataType;
  using Mode = nvinfer1::TensorIOMode;
  require(camera.engine->getNbIOTensors() == 3, "Engine A I/O count changed");
  expectTensor(*camera.engine, "images", Mode::kINPUT, Type::kFLOAT,
               {1, 6, 3, 256, 704});
  expectTensor(*camera.engine, "geometry", Mode::kINPUT, Type::kFLOAT,
               {1, 6, 59, 16, 44, 3});
  expectTensor(*camera.engine, "camera_bev", Mode::kOUTPUT, Type::kFLOAT,
               {1, 80, 180, 180});

  require(lidar.engine->getNbIOTensors() == 3, "Engine B I/O count changed");
  expectTensor(*lidar.engine, "points", Mode::kINPUT, Type::kFLOAT, {-1, 5});
  expectTensor(*lidar.engine, "lidar_bev", Mode::kOUTPUT, Type::kFLOAT,
               {1, 256, 180, 180});
  expectTensor(*lidar.engine, "lidar_status", Mode::kOUTPUT, Type::kINT32,
               {1});

  require(fusion.engine->getNbIOTensors() == 5, "Engine C I/O count changed");
  expectTensor(*fusion.engine, "camera_bev", Mode::kINPUT, Type::kFLOAT,
               {1, 80, 180, 180});
  expectTensor(*fusion.engine, "lidar_bev", Mode::kINPUT, Type::kFLOAT,
               {1, 256, 180, 180});
  expectTensor(*fusion.engine, "boxes", Mode::kOUTPUT, Type::kFLOAT,
               {1, 200, 9});
  expectTensor(*fusion.engine, "scores", Mode::kOUTPUT, Type::kFLOAT,
               {1, 200});
  expectTensor(*fusion.engine, "labels", Mode::kOUTPUT, Type::kINT32,
               {1, 200});
}

void bind(nvinfer1::IExecutionContext& context, char const* name,
          DeviceBuffer const& buffer) {
  require(context.setTensorAddress(name, buffer.data()),
          std::string("failed to bind ") + name);
}

void enqueue(nvinfer1::IExecutionContext& context, cudaStream_t stream,
             char const* stage) {
  require(context.enqueueV3(stream), std::string("enqueue failed for ") + stage);
}

std::size_t contextDeviceMemoryBytes(Engine const& engine) {
  return static_cast<std::size_t>(engine.engine->getDeviceMemorySizeV2());
}

}  // namespace thor_multisensor
