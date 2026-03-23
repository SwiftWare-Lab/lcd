
cmake_minimum_required(VERSION 3.18)
project(mycuda LANGUAGES CXX CUDA)

# ----------------------------
# CUDA architecture (compute_61 exactly like your nvcc)
# ----------------------------
set(CMAKE_CUDA_ARCHITECTURES 61)
# ----------------------------
# General settings
# ----------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# ----------------------------
# Compiler flags (match your nvcc command)
# ----------------------------
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -w -m64")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-dlcm=cg")

# Keep PTX (compute_61)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -gencode=arch=compute_61,code=sm_61 \
    -gencode=arch=compute_61,code=compute_61")

# ----------------------------
# Find CUDA toolkit (10.2)
# ----------------------------
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.2)
find_package(CUDAToolkit REQUIRED)

# ----------------------------
# Find pybind11
# ----------------------------
find_package(pybind11 REQUIRED)

# ----------------------------
# Create Python module (shared library)
# ----------------------------
pybind11_add_module(yysptrsv_wrapper pybind_yysptrsv.cu)

# Enable separable compilation (needed for kernels)
set_target_properties(yysptrsv_wrapper PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# ----------------------------
# Include directories
# ----------------------------
target_include_directories(yysptrsv_wrapper PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)

# ----------------------------
# Link libraries (match -lcusparse -lcudart)
# ----------------------------
target_link_libraries(yysptrsv_wrapper PRIVATE
        CUDA::cudart
        CUDA::cusparse
)