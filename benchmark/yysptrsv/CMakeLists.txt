
cmake_minimum_required(VERSION 3.18)
project(mycuda LANGUAGES CXX CUDA)

# GPU arch: pass -DCMAKE_CUDA_ARCHITECTURES=89 (or native) from the build script.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -w -m64")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-dlcm=cg")

if(DEFINED ENV{CUDA_HOME} AND NOT "$ENV{CUDA_HOME}" STREQUAL "")
    set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "CUDA root")
elseif(EXISTS /usr/local/cuda)
    set(CUDAToolkit_ROOT /usr/local/cuda CACHE PATH "CUDA root")
endif()

find_package(CUDAToolkit REQUIRED)
# With -DPython3_EXECUTABLE / INCLUDE_DIR / LIBRARY from build_external_modules.sh, venv + Ubuntu work without guesswork.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(yysptrsv_wrapper pybind_yysptrsv.cu)

set_target_properties(yysptrsv_wrapper PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(yysptrsv_wrapper PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(yysptrsv_wrapper PRIVATE
        CUDA::cudart
        CUDA::cusparse
)
