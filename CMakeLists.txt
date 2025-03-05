cmake_minimum_required(VERSION 3.15)

# Check if CMAKE_TOOLCHAIN_FILE is defined and exists before proceeding
if(DEFINED ENV{CMAKE_TOOLCHAIN_FILE})
    if(EXISTS "$ENV{CMAKE_TOOLCHAIN_FILE}")
        set(CMAKE_TOOLCHAIN_FILE "$ENV{CMAKE_TOOLCHAIN_FILE}")
        message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
    else()
        message(WARNING "Toolchain file specified by CMAKE_TOOLCHAIN_FILE environment variable does not exist: $ENV{CMAKE_TOOLCHAIN_FILE}")
        message(STATUS "Continuing without toolchain file...")
        unset(ENV{CMAKE_TOOLCHAIN_FILE})
    endif()
endif()

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
set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Path to Eigen3 include directory")

# Try different methods to find Eigen3
if(NOT EIGEN3_INCLUDE_DIR)
    # First try the standard find_package
    find_package(Eigen3 3.3 QUIET NO_MODULE)
    
    if(NOT Eigen3_FOUND)
        # Try finding the Eigen3Config.cmake file
        find_package(Eigen3 3.3 QUIET CONFIG)
    endif()
    
    if(NOT Eigen3_FOUND)
        # Try pkg-config
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(EIGEN3 QUIET eigen3)
        endif()
    endif()
    
    # If still not found, try to find the include directory directly
    if(NOT Eigen3_FOUND AND NOT EIGEN3_INCLUDE_DIR)
        find_path(EIGEN3_INCLUDE_DIR
            NAMES Eigen/Core
            PATHS
                ${CMAKE_INSTALL_PREFIX}/include
                ${CMAKE_PREFIX_PATH}/include
                /usr/include
                /usr/local/include
                /opt/local/include
                "C:/Program Files/Eigen3/include"
                "C:/Program Files (x86)/Eigen3/include"
                "C:/mingw64/include"
                "C:/msys64/mingw64/include"
                "$ENV{PROGRAMFILES}/Eigen3/include"
            PATH_SUFFIXES eigen3 eigen
        )
        
        if(EIGEN3_INCLUDE_DIR)
            message(STATUS "Found Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")
            add_library(Eigen3::Eigen INTERFACE IMPORTED)
            set_target_properties(Eigen3::Eigen PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
        else()
            message(FATAL_ERROR "Eigen3 not found. Please install Eigen3 or set EIGEN3_INCLUDE_DIR to the directory containing Eigen/Core")
        endif()
    endif()
endif()

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
    target_include_directories(config INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
endif()

if(BUILD_POLYREGRESSION)
    add_library(polyregression
        polyregression.cpp
        polyregression.h
    )
    target_include_directories(polyregression PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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
        kernels.cpp
        kernels.h
    )
    target_include_directories(gp-predict PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(BUILD_POLYREGRESSION)
    install(TARGETS polyregression
        EXPORT ${PROJECT_NAME}Targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(FILES polyregression.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

if(BUILD_GP_PREDICT)
    install(TARGETS gp-predict
        EXPORT ${PROJECT_NAME}Targets
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

# Generate and install package configuration files
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
