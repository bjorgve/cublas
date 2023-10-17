# CMake version
cmake_minimum_required(VERSION 3.11)

# Project's name
project(matrix_multiplication)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)

# Enable CUDA as a language
enable_language(CUDA)

# Find CUDA package
find_package(CUDA REQUIRED)

# Add CPU source file
add_library(matrix_mul_cpu src/matrix_mul_cpu.cpp)

# Add GPU source file
add_library(matrix_mul_gpu src/matrix_mul_gpu.cu)

# Set compiler options for GPU source
set_property(TARGET matrix_mul_gpu
             PROPERTY CUDA_ARCHITECTURES 60) # set this according to your GPU's compute capability
set_target_properties(matrix_mul_gpu
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Download Catch2
include(FetchContent)
FetchContent_Declare(
   Catch2
   GIT_REPOSITORY https://github.com/catchorg/Catch2.git
   GIT_TAG        v2.13.7
)
FetchContent_MakeAvailable(Catch2)

# GPU source requires CUBLAS
target_link_libraries(matrix_mul_gpu ${CUDA_cublas_LIBRARY})

# Add Catch2 main file
add_executable(catch_main tests/catch_main.cpp)

# Add main file
add_executable(main main.cpp)

# Specify target-specific include directories
target_include_directories(catch_main PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Link libraries to the executable
target_link_libraries(catch_main matrix_mul_cpu)
target_link_libraries(catch_main matrix_mul_gpu)
target_link_libraries(catch_main Catch2::Catch2)

# Link libraries to the executable
target_link_libraries(main matrix_mul_cpu)
target_link_libraries(main matrix_mul_gpu)
