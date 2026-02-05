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

#FetchContent_Declare(
#        nvbench
#        GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
#        GIT_TAG main
#)
#
#FetchContent_MakeAvailable(nvbench)


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
#        nvbench::main
        CUDA::cudart
        CUDA::cublas          # 🟩 required
        CUDA::cublasLt        # optional but recommended
        CUDA::cuda_driver
        CUDA::cusparse)

find_package(MPI)

# Check for NCCL
find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS /usr/local/cuda/include /usr/include
)
find_library(NCCL_LIBRARY cncl nccl
    HINTS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
)

if (NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
else()
    message(WARNING "NCCL not found! Multi-GPU features might fail.")
endif()

add_executable(multi_gpu_test multi_gpu_test.cu)
target_include_directories(multi_gpu_test PRIVATE ${eigen_SOURCE_DIR} ${NCCL_INCLUDE_DIR})
if(MPI_FOUND)
    target_link_libraries(multi_gpu_test PRIVATE MPI::MPI_CXX)
endif()
target_link_libraries(multi_gpu_test PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    ${NCCL_LIBRARY}
)




add_executable(test test_kernels.cu)
target_include_directories(test PRIVATE ${eigen_SOURCE_DIR})
target_link_libraries(test PRIVATE
        CUDA::cudart
        CUDA::cublas          # 🟩 required
        CUDA::cublasLt        # optional but recommended
        CUDA::cuda_driver
        CUDA::cusparse)