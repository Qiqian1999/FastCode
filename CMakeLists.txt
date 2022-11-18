cmake_minimum_required(VERSION 3.23)
project(FastCode)

set(CMAKE_CXX_STANDARD 14)

add_executable(FastCode main.cpp Filters.h Filters.cpp Matrix.h Matrix.cpp Tensor.h Tensor.cpp Utility.h Utility.cpp)
