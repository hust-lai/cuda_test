#include "xt_cuda.h"

bool __checkCudaDriver(CUresult e, const char* call, int line, const char *file) {
	if (e != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA Driver error %s, code = %d in file %s:%d\n", call, e, file, line);
		return false;
	}
	return true;
}

bool __checkCudaRuntime(cudaError_t e, const char* call, int line, const char *file) {
	if (e != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime error %s, code = %d, %s, %s in file %s:%d\n",
			call, e, cudaGetErrorName(e), cudaGetErrorString(e), file, line);
		return false;
	}
	return true;
}
