cmake_minimum_required(VERSION 3.15)

# Download and configure Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Add test executables
if(BUILD_CONFIG)
    add_executable(config_test config_test.cpp)
    target_link_libraries(config_test PRIVATE config gtest_main)
    add_test(NAME ConfigTest COMMAND config_test)
endif()

if(BUILD_POLYREGRESSION)
    add_executable(polyregression_test polyregression_test.cpp)
    target_link_libraries(polyregression_test PRIVATE polyregression gtest_main)
    add_test(NAME PolyRegressionTest COMMAND polyregression_test)
endif()

if(BUILD_GP_PREDICT)
    add_executable(gp_test gp_test.cpp)
    target_link_libraries(gp_test PRIVATE gp-predict gtest_main)
    add_test(NAME GPTest COMMAND gp_test)
    
    add_executable(kernels_test kernels_test.cpp)
    target_link_libraries(kernels_test PRIVATE gp-predict gtest_main)
    add_test(NAME KernelsTest COMMAND kernels_test)
endif()
