cmake_minimum_required(VERSION 3.18)
project(thor_dsvt_trt_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(THOR_CUDA_ARCHITECTURES "110" CACHE STRING
    "CUDA architectures used for the TensorRT plugins")
set(CMAKE_CUDA_ARCHITECTURES "${THOR_CUDA_ARCHITECTURES}")
set(DYNAMIC_BEVFUSION_MAX_POINTS "100000" CACHE STRING
    "Maximum raw points accepted by Engine B")
set(DYNAMIC_BEVFUSION_MAX_PILLARS "10000" CACHE STRING
    "Maximum occupied pillars retained by Engine B")
set(DYNAMIC_BEVFUSION_MAX_SETS "512" CACHE STRING
    "Maximum DSVT sets retained per shift")
if(DYNAMIC_BEVFUSION_MAX_POINTS LESS 1)
  message(FATAL_ERROR "DYNAMIC_BEVFUSION_MAX_POINTS must be positive")
endif()
if(DYNAMIC_BEVFUSION_MAX_PILLARS LESS 1 OR
   DYNAMIC_BEVFUSION_MAX_PILLARS GREATER 129600)
  message(FATAL_ERROR "DYNAMIC_BEVFUSION_MAX_PILLARS must be in [1,129600]")
endif()
if(DYNAMIC_BEVFUSION_MAX_SETS LESS 1 OR DYNAMIC_BEVFUSION_MAX_SETS GREATER 1690)
  message(FATAL_ERROR "DYNAMIC_BEVFUSION_MAX_SETS must be in [1,1690]")
endif()


find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS /usr/include/aarch64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY nvinfer
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR
   NOT TENSORRT_PLUGIN_LIBRARY)
  message(FATAL_ERROR "TensorRT headers or libraries were not found")
endif()

function(add_thor_trt_plugin target source)
  add_library(${target} SHARED ${source})
  target_include_directories(${target} PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE
    ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY} CUDA::cudart)
  target_compile_definitions(${target} PRIVATE
    DYNAMIC_BEVFUSION_MAX_POINTS=${DYNAMIC_BEVFUSION_MAX_POINTS}
    DYNAMIC_BEVFUSION_MAX_PILLARS=${DYNAMIC_BEVFUSION_MAX_PILLARS}
    DYNAMIC_BEVFUSION_MAX_SETS=${DYNAMIC_BEVFUSION_MAX_SETS})
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    BUILD_RPATH_USE_ORIGIN ON)
endfunction()

add_thor_trt_plugin(dynamic_pillar_decorate
  dynamic_pillar_decorate_plugin.cu)
add_thor_trt_plugin(dynamic_scatter_max
  scatter_max_plugin.cu)
add_thor_trt_plugin(dsvt_rotated_set
  rotated_set_plugin.cu)
add_thor_trt_plugin(tensor_barrier
  tensor_barrier_plugin.cu)
add_thor_trt_plugin(dsvt_dense_scatter
  dense_pillar_scatter_plugin.cu)

install(TARGETS
  dynamic_pillar_decorate
  dynamic_scatter_max
  dsvt_rotated_set
  tensor_barrier
  dsvt_dense_scatter
  LIBRARY DESTINATION .)
