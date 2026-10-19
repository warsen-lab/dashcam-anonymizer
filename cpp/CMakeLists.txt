cmake_minimum_required(VERSION 3.24)
project(video_compliance LANGUAGES CXX)

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

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(FFmpeg REQUIRED)
include(ThirdParty)

add_library(vc_core STATIC
  src/common/vc_error.cpp
  src/detect/box.cpp
  src/detect/letterbox.cpp
  src/io/frame_file_name.cpp
  src/io/jpeg_writer.cpp
  src/io/video_reader.cpp
  src/io/video_writer.cpp
  src/mask/mosaic.cpp
  src/sampler/frame_sampler.cpp
  src/sampler/luma_thumb.cpp
  src/track/box_timeline.cpp
)
target_include_directories(vc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(vc_core PUBLIC FFmpeg::FFmpeg)

add_library(vc_detect_lib STATIC
  src/detect/fake_detector.cpp
  src/detect/merged_detector.cpp
  src/detect/model_info.cpp
  src/detect/onnx_yolo_detector.cpp
  src/detect/ort_session.cpp
  src/detect/yolo_decode.cpp
  src/detect/yunet_decode.cpp
  src/detect/yunet_detector.cpp
)
# Only the .cpp files of this library include nlohmann/json; its headers do not.
target_link_libraries(vc_detect_lib PUBLIC vc_core vc_onnxruntime PRIVATE vc_nlohmann_json)

add_library(vc_extract_lib STATIC
  apps/vc_extract/extract_cli.cpp
  apps/vc_extract/extract_job.cpp
)
target_include_directories(vc_extract_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/apps")
target_link_libraries(vc_extract_lib PUBLIC vc_core)

find_package(Threads REQUIRED)

add_library(vc_redact_lib STATIC
  apps/vc_detect/detect_cli.cpp
  apps/vc_detect/detect_images.cpp
  apps/vc_redact/records.cpp
  apps/vc_redact/redact_cli.cpp
  apps/vc_redact/redact_config.cpp
  apps/vc_redact/redact_job.cpp
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
