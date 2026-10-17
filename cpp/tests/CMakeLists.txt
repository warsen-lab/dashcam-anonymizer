find_package(GTest REQUIRED)
find_program(VC_FFMPEG_EXECUTABLE ffmpeg REQUIRED)

add_executable(vc_tests
  test_media.cpp
  frame_sampler_test.cpp
  luma_thumb_test.cpp
  video_reader_test.cpp
)
target_link_libraries(vc_tests PRIVATE vc_core GTest::gtest_main)
target_compile_definitions(vc_tests PRIVATE
  VC_FFMPEG="${VC_FFMPEG_EXECUTABLE}"
  VC_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/tmp")

include(GoogleTest)
gtest_discover_tests(vc_tests)
