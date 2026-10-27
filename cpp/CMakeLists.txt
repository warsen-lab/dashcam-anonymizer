cmake_minimum_required(VERSION 3.24)
project(dashcam_anonymizer LANGUAGES CXX)

# Single-config generators (Makefiles, Ninja) default to an unoptimised build otherwise.
get_property(_vc_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _vc_multi_config AND NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMAKE_BUILD_TYPE is not set, defaulting to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# The GPU code paths (spec 2026-10-20): CUDA kernels for NVIDIA cards. OFF (the default, macOS and every local
# test) compiles and links no CUDA at all; the GPU code is behind #ifdef VC_WITH_CUDA.
option(VC_WITH_CUDA "Build the CUDA code paths for NVIDIA GPUs (needs the CUDA 12 toolkit)" OFF)
if(VC_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "VC_WITH_CUDA=ON needs the CUDA toolkit (nvcc), which CMake did not find: install CUDA 12 "
                        "(scripts/setup_linux.sh) or configure with -DVC_WITH_CUDA=OFF")
  endif()
  # 86: A10 / RTX 3090, 89: L4 / L40S / RTX 4090 (spec 2026-10-20 §2).
  set(CMAKE_CUDA_ARCHITECTURES "86;89" CACHE STRING "CUDA architectures to compile the kernels for")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_compile_definitions(VC_WITH_CUDA=1)
  message(STATUS "VC_WITH_CUDA: ON (CUDA architectures ${CMAKE_CUDA_ARCHITECTURES})")
endif()

# C++ only: nvcc takes host-compiler warnings through -Xcompiler, which the CUDA sources set themselves.
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")

find_package(FFmpeg REQUIRED)
include(ThirdParty)

add_library(vc_core STATIC
  src/common/vc_error.cpp
  src/detect/box.cpp
  src/detect/letterbox.cpp
  src/detect/verifier_crop.cpp
  src/io/frame_file_name.cpp
  src/io/jpeg_writer.cpp
  src/io/video_reader.cpp
  src/io/video_writer.cpp
  src/mask/mosaic.cpp
  src/sampler/frame_sampler.cpp
  src/sampler/luma_thumb.cpp
  src/track/box_matching.cpp
  src/track/box_timeline.cpp
  src/track/keyframe_linker.cpp
  src/track/track_masker.cpp
)
target_include_directories(vc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(vc_core PUBLIC FFmpeg::FFmpeg)

find_package(Threads REQUIRED)

add_library(vc_detect_lib STATIC
  src/detect/fake_detector.cpp
  src/detect/merged_detector.cpp
  src/detect/model_info.cpp
  src/detect/onnx_yolo_detector.cpp
  src/detect/ort_session.cpp
  src/detect/yolo_decode.cpp
  src/detect/yunet_decode.cpp
  src/detect/yunet_detector.cpp
  src/detect/verifier.cpp
)
# Only the .cpp files of this library include nlohmann/json; its headers do not.
target_link_libraries(vc_detect_lib PUBLIC vc_core vc_onnxruntime Threads::Threads PRIVATE vc_nlohmann_json)

add_library(vc_extract_lib STATIC
  apps/vc_extract/extract_cli.cpp
  apps/vc_extract/extract_job.cpp
)
target_include_directories(vc_extract_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/apps")
target_link_libraries(vc_extract_lib PUBLIC vc_core)

add_library(vc_redact_lib STATIC
  apps/vc_detect/detect_cli.cpp
  apps/vc_detect/detect_images.cpp
  apps/vc_redact/detect_rate.cpp
  apps/vc_redact/records.cpp
  apps/vc_redact/redact_cli.cpp
  apps/vc_redact/redact_config.cpp
  apps/vc_redact/redact_job.cpp
  apps/vc_redact/verify_dump.cpp
  apps/vc_redact/verify_policy.cpp
)
target_include_directories(vc_redact_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/apps")
# redact_config.h includes nlohmann/json, so it is part of this library's interface.
target_link_libraries(vc_redact_lib PUBLIC vc_detect_lib vc_nlohmann_json Threads::Threads)

add_executable(vc_extract apps/vc_extract/main.cpp)
target_link_libraries(vc_extract PRIVATE vc_extract_lib)

add_executable(vc_redact apps/vc_redact/main.cpp)
target_link_libraries(vc_redact PRIVATE vc_redact_lib)

add_executable(vc_detect apps/vc_detect/main.cpp)
target_link_libraries(vc_detect PRIVATE vc_redact_lib)

option(VC_BUILD_TESTS "Build the GoogleTest suite" ON)
if(VC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
