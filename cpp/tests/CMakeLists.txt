find_package(GTest REQUIRED)
find_program(VC_FFMPEG_EXECUTABLE ffmpeg REQUIRED)

add_executable(vc_tests
  test_media.cpp
  box_test.cpp
  box_timeline_test.cpp
  build_config_test.cpp
  detect_rate_test.cpp
  bounded_queue_test.cpp
  detect_cli_test.cpp
  detect_images_test.cpp
  execution_provider_test.cpp
  extract_cli_test.cpp
  extract_job_test.cpp
  fake_detector_test.cpp
  frame_file_name_test.cpp
  frame_sampler_test.cpp
  jpeg_writer_test.cpp
  letterbox_test.cpp
  luma_thumb_test.cpp
  merged_detector_test.cpp
  model_info_test.cpp
  mosaic_grid_test.cpp
  mosaic_test.cpp
  onnx_yolo_detector_test.cpp
  records_test.cpp
  redact_cli_test.cpp
  redact_config_test.cpp
  redact_job_test.cpp
  video_reader_test.cpp
  verifier_crop_test.cpp
  verifier_test.cpp
  verify_policy_test.cpp
  video_writer_test.cpp
  yolo_decode_test.cpp
  yunet_decode_test.cpp
  yunet_detector_test.cpp
)
target_link_libraries(vc_tests PRIVATE vc_extract_lib vc_redact_lib vc_detect_lib vc_core GTest::gtest_main)
target_compile_definitions(vc_tests PRIVATE
  VC_FFMPEG="${VC_FFMPEG_EXECUTABLE}"
  VC_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  VC_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/tmp"
  VC_REPO_DIR="${PROJECT_SOURCE_DIR}/.."
  VC_TEST_EXPECT_CUDA=$<BOOL:${VC_WITH_CUDA}>
)

include(GoogleTest)
# A test that hangs (e.g. on a FIFO) is killed by ctest instead of blocking forever.
gtest_discover_tests(vc_tests PROPERTIES TIMEOUT 120)
