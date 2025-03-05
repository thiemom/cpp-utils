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
install(TARGETS 
    $<$<BOOL:${BUILD_POLYREGRESSION}>:polyregression_example>
    $<$<BOOL:${BUILD_GP_PREDICT}>:gp_example>
    $<$<BOOL:${BUILD_GP_PREDICT}>:composite_kernel_example>
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
