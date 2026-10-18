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
set(VC_ONNXRUNTIME_ROOT "${VC_THIRD_PARTY_DIR}/onnxruntime-osx-arm64-1.30.0"
    CACHE PATH "Unpacked ONNX Runtime package (include/, lib/)")

if(NOT EXISTS "${VC_NLOHMANN_JSON_INCLUDE}/nlohmann/json.hpp")
  message(FATAL_ERROR
    "nlohmann/json.hpp not found under VC_NLOHMANN_JSON_INCLUDE=${VC_NLOHMANN_JSON_INCLUDE} (see cpp/README.md)")
endif()
add_library(vc_nlohmann_json INTERFACE)
target_include_directories(vc_nlohmann_json SYSTEM INTERFACE "${VC_NLOHMANN_JSON_INCLUDE}")

find_library(VC_ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${VC_ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT VC_ONNXRUNTIME_LIBRARY OR NOT EXISTS "${VC_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ONNX Runtime not found under VC_ONNXRUNTIME_ROOT=${VC_ONNXRUNTIME_ROOT} (see cpp/README.md)")
endif()
add_library(vc_onnxruntime SHARED IMPORTED)
set_target_properties(vc_onnxruntime PROPERTIES
  IMPORTED_LOCATION "${VC_ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${VC_ONNXRUNTIME_ROOT}/include"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${VC_ONNXRUNTIME_ROOT}/include")
# libonnxruntime's install name is @rpath/libonnxruntime.1.dylib.
list(APPEND CMAKE_BUILD_RPATH "${VC_ONNXRUNTIME_ROOT}/lib")
