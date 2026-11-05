function(minitensor_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE minitensor::minitensor)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)

    # Every test runs on each set of SIMD kernels. "best" is whatever the CPU
    # supports, which is AVX-512 where available.
    add_test(NAME ${name}.best COMMAND ${name})
    add_test(NAME ${name}.generic COMMAND ${name})
    set_tests_properties(${name}.generic PROPERTIES ENVIRONMENT "MINITENSOR_ISA=generic")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        add_test(NAME ${name}.avx2 COMMAND ${name})
        set_tests_properties(${name}.avx2 PROPERTIES ENVIRONMENT "MINITENSOR_ISA=avx2")
    endif()
endfunction()

minitensor_add_test(test_creation)
minitensor_add_test(test_indexing)
minitensor_add_test(test_elementwise)
minitensor_add_test(test_matmul)
minitensor_add_test(test_utilities)

# Timings only mean something in an optimized build, running on its own.
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT MINITENSOR_SANITIZE)
    add_executable(test_efficiency test_efficiency.cpp)
    target_link_libraries(test_efficiency PRIVATE minitensor::minitensor)
    target_include_directories(test_efficiency PRIVATE ${PROJECT_SOURCE_DIR}/benchmarks)
    target_compile_options(test_efficiency PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME test_efficiency COMMAND test_efficiency)
    set_tests_properties(test_efficiency PROPERTIES RUN_SERIAL TRUE)
endif()

# Tiny cache sizes force matrices to be split across every level of blocking.
add_test(NAME test_matmul.small_caches COMMAND test_matmul)
set_tests_properties(test_matmul.small_caches PROPERTIES ENVIRONMENT
    "MINITENSOR_L1_CACHE=4096;MINITENSOR_L2_CACHE=32768;MINITENSOR_L3_CACHE=131072")
