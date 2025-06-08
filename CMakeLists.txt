cmake_minimum_required(VERSION 3.30)

set(CMAKE_CUDA_COMPILER /home/xusj/conda/envs/gsy_cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /home/xusj/conda/envs/gsy_cuda/bin/clang++)
set(CMAKE_CUDA_ARCHITECTURES 70)

project(cu_utility LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB cu_utility_src ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
add_library(cu_utility SHARED ${cu_utility_src})
target_include_directories(
    cu_utility PUBLIC     
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(cu_utility PUBLIC --expt-relaxed-constexpr)

add_subdirectory(example)

install(
    TARGETS cu_utility
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cu_utilityConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cu_utilityConfig.cmake
    INSTALL_DESTINATION lib/cmake/cu_utility
)

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cu_utilityConfig.cmake
    DESTINATION lib/cmake/cu_utility
)

install(
    TARGETS cu_utility
    EXPORT cu_utilityTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)

install(
    EXPORT cu_utilityTargets
    NAMESPACE cu_utility::
    DESTINATION lib/cmake/cu_utility
)