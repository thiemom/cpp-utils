cmake_minimum_required(VERSION 3.15)
project(cpp_ml_tools VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options for component selection
option(BUILD_POLYREGRESSION "Build polynomial regression component" ON)
option(BUILD_GP_PREDICT "Build Gaussian process prediction component" ON)
option(BUILD_CONFIG "Build configuration utility component" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test suite" OFF)

# Find Eigen (required for all components)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Platform-specific settings
if(WIN32)
    add_compile_definitions(_USE_MATH_DEFINES)
    if(MINGW)
        add_compile_options(-Wall -Wextra)
    endif()
else()
    add_compile_options(-Wall -Wextra)
endif()

# Add components based on options
if(BUILD_CONFIG)
    add_library(config INTERFACE)
    target_include_directories(config INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(BUILD_POLYREGRESSION)
    add_library(polyregression
        polyregression.cpp
        polyregression.h
    )
    target_link_libraries(polyregression PUBLIC Eigen3::Eigen)
    if(BUILD_CONFIG)
        target_link_libraries(polyregression PUBLIC config)
    endif()
endif()

if(BUILD_GP_PREDICT)
    add_library(gp-predict
        gp-predict.cpp
        gp-predict.h
        kernels.h
    )
    target_link_libraries(gp-predict PUBLIC Eigen3::Eigen)
    if(BUILD_CONFIG)
        target_link_libraries(gp-predict PUBLIC config)
    endif()
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Installation rules
include(GNUInstallDirs)
install(TARGETS config
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(BUILD_POLYREGRESSION)
    install(TARGETS polyregression
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(FILES polyregression.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

if(BUILD_GP_PREDICT)
    install(TARGETS gp-predict
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(FILES 
        gp-predict.h 
        kernels.h 
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

# Export targets
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
