//
// Created by salehm32 on 2025-11-17.
//

#pragma once
#include <cuda_runtime.h>

__global__ void add_kernel(float* a, const float* b, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a[i] += b[i];
}