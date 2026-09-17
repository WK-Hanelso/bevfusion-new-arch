cmake_minimum_required(VERSION 3.18)
project(bevfusion_trt_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS /usr/include/aarch64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY nvinfer
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_PLUGIN_LIBRARY)
  message(FATAL_ERROR "TensorRT headers or libraries were not found")
endif()

add_executable(bevfusion_trt_infer
  main.cpp
  common.cpp
  options.cpp
  point_features.cpp
  cuda_resources.cpp
  trt_components.cpp
  statistics.cpp
  multisensor_pipeline.cpp)
target_include_directories(bevfusion_trt_infer PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(bevfusion_trt_infer PRIVATE
  ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY} CUDA::cudart
  ${CMAKE_DL_LIBS} Threads::Threads)
target_compile_options(bevfusion_trt_infer PRIVATE -Wall -Wextra -Wpedantic)
