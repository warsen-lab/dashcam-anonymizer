cmake_minimum_required(VERSION 3.24)
project(video_compliance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(FFmpeg REQUIRED)

add_library(vc_core STATIC
  src/common/vc_error.cpp
  src/io/jpeg_writer.cpp
  src/io/video_reader.cpp
  src/sampler/frame_sampler.cpp
  src/sampler/luma_thumb.cpp
)
target_include_directories(vc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(vc_core PUBLIC FFmpeg::FFmpeg)

add_library(vc_extract_lib STATIC
  apps/vc_extract/extract_cli.cpp
  apps/vc_extract/extract_job.cpp
)
target_include_directories(vc_extract_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/apps")
target_link_libraries(vc_extract_lib PUBLIC vc_core)

add_executable(vc_extract apps/vc_extract/main.cpp)
target_link_libraries(vc_extract PRIVATE vc_extract_lib)

option(VC_BUILD_TESTS "Build the GoogleTest suite" ON)
if(VC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
