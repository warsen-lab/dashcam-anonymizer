cmake_minimum_required(VERSION 3.24)
project(video_compliance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(vc_core STATIC
  src/sampler/frame_sampler.cpp
)
target_include_directories(vc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

option(VC_BUILD_TESTS "Build the GoogleTest suite" ON)
if(VC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
