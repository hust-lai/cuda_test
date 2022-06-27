#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define checkDriver(call)            __checkCudaDriver(call, #call, __LINE__, __FILE__)
#define checkRuntime(call)           __checkCudaRuntime(call, #call, __LINE__, __FILE__)

#define checkKernel(...)                                                                             \
    __VA_ARGS__;                                                                                     \
    do{cudaError_t cudaStatus = cudaPeekAtLastError();                                               \
    if (cudaStatus != cudaSuccess){                                                                  \
        fprintf(stderr, "Launch kernel failed: %s in file %s:%d\n", cudaGetErrorString(cudaStatus), __FILE__, __LINE__);  \
    }} while(0);

bool __checkCudaDriver(CUresult e, const char* call, int line, const char *file);
bool __checkCudaRuntime(cudaError_t e, const char* call, int line, const char *file);
