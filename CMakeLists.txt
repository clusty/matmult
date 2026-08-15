cmake_minimum_required(VERSION 3.20)
project(matmult)

set(CMAKE_CXX_STANDARD 26)

add_executable(matmult main.cpp)

target_compile_options(matmult PRIVATE -Ofast -march=native -fopenmp)
target_link_options(matmult PRIVATE -Ofast -march=native -fopenmp)