cmake_minimum_required(VERSION 3.18)

project(cusparse_sptrsv_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Change this if your source file has a different name
set(SRC_FILE main.cpp)

add_executable(sptrsv_bench ${SRC_FILE})

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(sptrsv_bench PRIVATE CUDA::cusparse CUDA::cudart)

# Optional but recommended
set_target_properties(sptrsv_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)

# Set the GPU architectures you want to build for.
# Adjust as needed for your machine.
# Examples:
#   75 = Turing
#   80 = A100
#   86 = RTX 30xx
#   89 = RTX 40xx Ada
#   90 = H100
set_target_properties(sptrsv_bench PROPERTIES
        CUDA_ARCHITECTURES 86
)

# Good optimization defaults
target_compile_options(sptrsv_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

pybind11_add_module(cusparse_sptrsv MODULE cusparseWrapper.cpp)

target_link_libraries(cusparse_sptrsv
        PRIVATE
        CUDA::cudart
        CUDA::cusparse
)

set_target_properties(cusparse_sptrsv PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES 86
)

target_compile_options(cusparse_sptrsv PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)