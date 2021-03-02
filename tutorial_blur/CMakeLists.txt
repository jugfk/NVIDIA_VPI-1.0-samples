cmake_minimum_required(VERSION 3.5)

project(vpi_blur)

# This instructs cmake to look for the most recent
# vpi instance installed on the system.
find_package(vpi REQUIRED)
find_package(OpenCV REQUIRED)

# Creates the blur executable target
add_executable(vpi_blur main.cpp)

# It uses vpi and opencv. CMake will automatically
# set up the correct header and library directories,
# and make hello_work link to these libraries.
target_link_libraries(vpi_blur vpi opencv_core)

# OpenCV < 3 uses a different library for image i/o
if(OpenCV_VERSION VERSION_LESS 3)
    target_link_libraries(vpi_blur opencv_highgui)
else()
    target_link_libraries(vpi_blur opencv_imgcodecs)
endif()
