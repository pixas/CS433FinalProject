cmake_minimum_required(VERSION 2.8)

project(oracle)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(oracle oracle.cpp)

target_link_libraries(oracle ${OpenCV_LIBS})