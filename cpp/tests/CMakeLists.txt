find_package(GTest REQUIRED)

add_executable(vc_tests
  frame_sampler_test.cpp
)
target_link_libraries(vc_tests PRIVATE vc_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(vc_tests)
