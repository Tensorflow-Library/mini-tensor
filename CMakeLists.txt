cmake_minimum_required(VERSION 3.16)

project(minitensor
    VERSION 1.0.0
    DESCRIPTION "A small, fast tensor library written from scratch in C++17"
    HOMEPAGE_URL "https://github.com/Tensorflow-Library/mini-tensor"
    LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MINITENSOR_TOP_LEVEL ON)
else()
    set(MINITENSOR_TOP_LEVEL OFF)
endif()

option(MINITENSOR_OPENMP "Use OpenMP for multithreading" ON)
option(MINITENSOR_BUILD_TESTS "Build the tests" ${MINITENSOR_TOP_LEVEL})
option(MINITENSOR_BUILD_EXAMPLES "Build the examples" ${MINITENSOR_TOP_LEVEL})
option(MINITENSOR_BUILD_BENCHMARKS "Build the benchmarks" ${MINITENSOR_TOP_LEVEL})
option(MINITENSOR_INSTALL "Generate the install target" ${MINITENSOR_TOP_LEVEL})
option(MINITENSOR_SANITIZE "Build with the address and undefined behaviour sanitizers" OFF)

if(MINITENSOR_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type chosen, using Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(minitensor
    src/kernels/kernels_generic.cpp
    src/matmul.cpp
    src/memory.cpp
    src/ops.cpp
    src/runtime.cpp)
add_library(minitensor::minitensor ALIAS minitensor)

target_compile_features(minitensor PUBLIC cxx_std_17)
target_include_directories(minitensor
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(minitensor PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})

if(MINITENSOR_TOP_LEVEL)
    target_compile_options(minitensor PRIVATE -Wall -Wextra -Wpedantic)
endif()

# The SIMD kernels are always optimized, so that a program built in Debug mode
# still gets fast tensor operations. `-fopenmp-simd` turns on `omp simd` loops
# even when OpenMP threads are off.
set(MINITENSOR_KERNEL_OPTIONS -O3 -fopenmp-simd -ffp-contract=fast)
set_source_files_properties(src/kernels/kernels_generic.cpp PROPERTIES
    COMPILE_OPTIONS "${MINITENSOR_KERNEL_OPTIONS}")

# On x86-64, the same kernels are also built for newer CPUs and the best
# version is chosen when the program starts (see src/runtime.cpp).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_sources(minitensor PRIVATE
        src/kernels/kernels_avx2.cpp
        src/kernels/kernels_avx512.cpp)
    target_compile_definitions(minitensor PRIVATE MINITENSOR_HAVE_AVX2 MINITENSOR_HAVE_AVX512)
    set_source_files_properties(src/kernels/kernels_avx2.cpp PROPERTIES
        COMPILE_OPTIONS "${MINITENSOR_KERNEL_OPTIONS};-mavx2;-mfma")
    set_source_files_properties(src/kernels/kernels_avx512.cpp PROPERTIES
        COMPILE_OPTIONS
        "${MINITENSOR_KERNEL_OPTIONS};-mavx512f;-mavx512dq;-mavx512vl;-mfma;-mprefer-vector-width=512")
endif()

set(MINITENSOR_USES_OPENMP OFF)
if(MINITENSOR_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(minitensor PUBLIC OpenMP::OpenMP_CXX)
        set(MINITENSOR_USES_OPENMP ON)
    else()
        message(WARNING "OpenMP was not found, so minitensor will run on one thread")
    endif()
endif()
if(NOT MINITENSOR_USES_OPENMP)
    target_compile_options(minitensor PUBLIC -Wno-unknown-pragmas)
endif()

if(MINITENSOR_SANITIZE)
    target_compile_options(minitensor PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(minitensor PUBLIC -fsanitize=address,undefined)
endif()

if(MINITENSOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(MINITENSOR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(MINITENSOR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(MINITENSOR_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS minitensor EXPORT minitensorTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT minitensorTargets
        NAMESPACE minitensor::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minitensor)

    configure_package_config_file(cmake/minitensorConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/minitensorConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minitensor)
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/minitensorConfigVersion.cmake
        COMPATIBILITY SameMajorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/minitensorConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/minitensorConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minitensor)
endif()
