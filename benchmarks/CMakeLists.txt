add_executable(bench_tensor bench_tensor.cpp)
target_link_libraries(bench_tensor PRIVATE minitensor::minitensor)
