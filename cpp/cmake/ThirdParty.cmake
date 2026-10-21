# nlohmann/json (header only) and the ONNX Runtime prebuilt package, unpacked once into the cache
# shared by every checkout (see cpp/README.md). Nothing is downloaded at configure time.
#
# Defines:
#   vc_nlohmann_json  INTERFACE target with the json.hpp include directory
#   vc_onnxruntime    IMPORTED SHARED target for libonnxruntime
#
# The package's own lib/cmake/onnxruntime config points at a non-existent include/onnxruntime
# directory, so the imported target is defined here instead.

set(VC_THIRD_PARTY_DIR "$ENV{HOME}/.cache/video-compliance/third_party"
    CACHE PATH "Directory holding the unpacked third-party packages")
set(VC_NLOHMANN_JSON_INCLUDE "${VC_THIRD_PARTY_DIR}/nlohmann_json-3.12.0/include"
    CACHE PATH "Directory containing nlohmann/json.hpp")
# The package for the platform: macOS arm64 (CoreML EP), Linux x64 GPU (CUDA and TensorRT EPs, CPU as well).
if(APPLE)
  set(_vc_onnxruntime_package "onnxruntime-osx-arm64-1.30.0")
else()
  set(_vc_onnxruntime_package "onnxruntime-linux-x64-gpu-1.30.0")
endif()
set(VC_ONNXRUNTIME_ROOT "${VC_THIRD_PARTY_DIR}/${_vc_onnxruntime_package}"
    CACHE PATH "Unpacked ONNX Runtime package (include/, lib/)")

if(NOT EXISTS "${VC_NLOHMANN_JSON_INCLUDE}/nlohmann/json.hpp")
  message(FATAL_ERROR
    "nlohmann/json.hpp not found under VC_NLOHMANN_JSON_INCLUDE=${VC_NLOHMANN_JSON_INCLUDE} (see cpp/README.md)")
endif()
add_library(vc_nlohmann_json INTERFACE)
target_include_directories(vc_nlohmann_json SYSTEM INTERFACE "${VC_NLOHMANN_JSON_INCLUDE}")

# NO_CACHE: a cached result would keep pointing at the old package after VC_ONNXRUNTIME_ROOT changes.
# The unset drops an entry that an older configure of this directory cached (find_library skips the
# search while the variable is set).
unset(VC_ONNXRUNTIME_LIBRARY CACHE)
find_library(VC_ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${VC_ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH NO_CACHE)
if(NOT VC_ONNXRUNTIME_LIBRARY OR NOT EXISTS "${VC_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ONNX Runtime not found under VC_ONNXRUNTIME_ROOT=${VC_ONNXRUNTIME_ROOT} (see cpp/README.md)")
endif()
add_library(vc_onnxruntime SHARED IMPORTED)
set_target_properties(vc_onnxruntime PROPERTIES
  IMPORTED_LOCATION "${VC_ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${VC_ONNXRUNTIME_ROOT}/include"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${VC_ONNXRUNTIME_ROOT}/include")
# libonnxruntime's install name is @rpath/libonnxruntime.1.dylib (on Linux the binaries find libonnxruntime.so.1
# through the same RPATH entry).
list(APPEND CMAKE_BUILD_RPATH "${VC_ONNXRUNTIME_ROOT}/lib")
