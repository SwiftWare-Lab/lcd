cmake_minimum_required(VERSION 3.20)
project(cudagraph_nvbench LANGUAGES CXX CUDA)

find_package(CUDA)
find_package(CUDAToolkit)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "-g -G")


# Enable modern CUDA features
set(CMAKE_CUDA_ARCHITECTURES 50 61 75 80 86)

# FetchContent NVBench
include(FetchContent)

FetchContent_Declare(
        nvbench
        GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
        GIT_TAG main
)

FetchContent_MakeAvailable(nvbench)


FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)

FetchContent_MakeAvailable(eigen)

# Build benchmark executable
add_executable(bench main.cu)
target_include_directories(bench PRIVATE ${eigen_SOURCE_DIR})
target_link_libraries(bench PRIVATE
        nvbench::main
        CUDA::cudart
        CUDA::cublas          # 🟩 required
        CUDA::cublasLt        # optional but recommended
        CUDA::cuda_driver)




add_executable(test test_kernels.cu)
target_include_directories(test PRIVATE ${eigen_SOURCE_DIR})
target_link_libraries(test PRIVATE
        CUDA::cudart
        CUDA::cublas          # 🟩 required
        CUDA::cublasLt        # optional but recommended
        CUDA::cuda_driver
        CUDA::cusparse)