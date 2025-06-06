cmake_minimum_required(VERSION 3.30)

set(CMAKE_CUDA_COMPILER /home/xusj/conda/envs/gsy_cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /home/xusj/conda/envs/gsy_cuda/bin/clang++)
set(CMAKE_CUDA_ARCHITECTURES 70)

project(cu_utility LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB cu_utility_src ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
add_library(cu_utility SHARED ${cu_utility_src})
target_include_directories(cu_utility PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(cu_utility PUBLIC --expt-relaxed-constexpr)

add_subdirectory(example)