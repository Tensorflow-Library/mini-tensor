foreach(example quickstart linear_regression neural_network image_blur)
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE minitensor::minitensor)
endforeach()
