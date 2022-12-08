cmake_minimum_required(VERSION 2.8)

project(utils)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(load_parameters load_parameters.cpp)

target_link_libraries(load_parameters ${OpenCV_LIBS})