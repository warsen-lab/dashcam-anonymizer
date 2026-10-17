# Finds libavformat, libavcodec, libavutil and libswscale without pkg-config.
# Defines the imported target FFmpeg::FFmpeg.

set(_ffmpeg_hints "")
if(APPLE)
  execute_process(
    COMMAND brew --prefix ffmpeg
    OUTPUT_VARIABLE _ffmpeg_brew_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_ffmpeg_brew_prefix)
    list(APPEND _ffmpeg_hints "${_ffmpeg_brew_prefix}")
  endif()
endif()

find_path(FFmpeg_INCLUDE_DIR libavcodec/avcodec.h HINTS ${_ffmpeg_hints} PATH_SUFFIXES include)

set(_ffmpeg_library_vars "")
foreach(_lib avformat avcodec swscale avutil)
  find_library(FFmpeg_${_lib}_LIBRARY NAMES ${_lib} HINTS ${_ffmpeg_hints} PATH_SUFFIXES lib)
  list(APPEND _ffmpeg_library_vars FFmpeg_${_lib}_LIBRARY)
endforeach()

mark_as_advanced(FFmpeg_INCLUDE_DIR ${_ffmpeg_library_vars})

# The code calls sws_scale_frame() with an uninitialised SwsContext, which needs the dynamic
# context of FFmpeg 8 (libswscale 9). Read the major version from the headers.
set(FFmpeg_SWSCALE_VERSION_MAJOR "")
if(FFmpeg_INCLUDE_DIR AND EXISTS "${FFmpeg_INCLUDE_DIR}/libswscale/version_major.h")
  file(STRINGS "${FFmpeg_INCLUDE_DIR}/libswscale/version_major.h" _swscale_major_line
       REGEX "^#define[ \t]+LIBSWSCALE_VERSION_MAJOR[ \t]+[0-9]+")
  if(_swscale_major_line MATCHES "LIBSWSCALE_VERSION_MAJOR[ \t]+([0-9]+)")
    set(FFmpeg_SWSCALE_VERSION_MAJOR "${CMAKE_MATCH_1}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFmpeg REQUIRED_VARS FFmpeg_INCLUDE_DIR ${_ffmpeg_library_vars})

if(FFmpeg_FOUND AND (FFmpeg_SWSCALE_VERSION_MAJOR STREQUAL "" OR FFmpeg_SWSCALE_VERSION_MAJOR LESS 9))
  if(FFmpeg_SWSCALE_VERSION_MAJOR STREQUAL "")
    set(_swscale_found "unknown libswscale version")
  else()
    set(_swscale_found "libswscale ${FFmpeg_SWSCALE_VERSION_MAJOR}")
  endif()
  set(_ffmpeg_old_message
      "FFmpeg >= 8 (libswscale >= 9) is required, found ${_swscale_found} in ${FFmpeg_INCLUDE_DIR}")
  if(FFmpeg_FIND_REQUIRED)
    message(FATAL_ERROR "${_ffmpeg_old_message}")
  endif()
  message(STATUS "${_ffmpeg_old_message}")
  set(FFmpeg_FOUND FALSE)
endif()

if(FFmpeg_FOUND AND NOT TARGET FFmpeg::FFmpeg)
  add_library(FFmpeg::FFmpeg INTERFACE IMPORTED)
  target_include_directories(FFmpeg::FFmpeg INTERFACE "${FFmpeg_INCLUDE_DIR}")
  target_link_libraries(FFmpeg::FFmpeg INTERFACE
    "${FFmpeg_avformat_LIBRARY}"
    "${FFmpeg_avcodec_LIBRARY}"
    "${FFmpeg_swscale_LIBRARY}"
    "${FFmpeg_avutil_LIBRARY}")
endif()
