# A separate project, showing how another program uses an installed minitensor.
#
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/install
#   cmake --build build
cmake_minimum_required(VERSION 3.16)
project(consumer LANGUAGES CXX)

find_package(minitensor 1.0 REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE minitensor::minitensor)
