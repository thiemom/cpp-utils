# Download and configure Google Test
include(FetchContent)

# Set CMake policies to avoid deprecation warnings
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0054 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0056 NEW)

# Configure GoogleTest with a newer minimum CMake version
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
  OVERRIDE_FIND_PACKAGE
  CMAKE_ARGS
    -DCMAKE_MINIMUM_REQUIRED_VERSION=3.10
)

# GoogleTest policies are set above

FetchContent_MakeAvailable(googletest)

# Add test executables
if(BUILD_CONFIG)
    add_executable(config_test config_test.cpp)
    target_link_libraries(config_test PRIVATE config gtest_main)
    add_test(NAME ConfigTest COMMAND $<TARGET_FILE:config_test>)
endif()

if(BUILD_POLYREGRESSION)
    add_executable(polyregression_test polyregression_test.cpp)
    target_link_libraries(polyregression_test PRIVATE polyregression gtest_main)
    add_test(NAME PolyRegressionTest COMMAND $<TARGET_FILE:polyregression_test>)
endif()

if(BUILD_GP_PREDICT)
    add_executable(gp_test gp_test.cpp)
    target_link_libraries(gp_test PRIVATE gp-predict gtest_main)
    add_test(NAME GPTest COMMAND $<TARGET_FILE:gp_test>)
    
    add_executable(kernels_test kernels_test.cpp)
    target_link_libraries(kernels_test PRIVATE gp-predict gtest_main)
    add_test(NAME KernelsTest COMMAND $<TARGET_FILE:kernels_test>)
endif()
