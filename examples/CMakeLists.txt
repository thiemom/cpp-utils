# Example programs

if(BUILD_POLYREGRESSION)
    add_executable(polyregression_example
        polyregression_usage.cpp
    )
    target_link_libraries(polyregression_example PRIVATE polyregression)
endif()

if(BUILD_GP_PREDICT)
    add_executable(gp_example
        gp_usage.cpp
    )
    target_link_libraries(gp_example PRIVATE gp-predict)

    add_executable(composite_kernel_example
        composite_kernel_example.cpp
    )
    target_link_libraries(composite_kernel_example PRIVATE gp-predict)
endif()

# Installation
if(BUILD_POLYREGRESSION)
    install(TARGETS polyregression_example
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(BUILD_GP_PREDICT)
    install(TARGETS gp_example composite_kernel_example
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
