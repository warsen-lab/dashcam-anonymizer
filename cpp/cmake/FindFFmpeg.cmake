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

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFmpeg REQUIRED_VARS FFmpeg_INCLUDE_DIR ${_ffmpeg_library_vars})

if(FFmpeg_FOUND AND NOT TARGET FFmpeg::FFmpeg)
  add_library(FFmpeg::FFmpeg INTERFACE IMPORTED)
  target_include_directories(FFmpeg::FFmpeg INTERFACE "${FFmpeg_INCLUDE_DIR}")
  target_link_libraries(FFmpeg::FFmpeg INTERFACE
    "${FFmpeg_avformat_LIBRARY}"
    "${FFmpeg_avcodec_LIBRARY}"
    "${FFmpeg_swscale_LIBRARY}"
    "${FFmpeg_avutil_LIBRARY}")
endif()
