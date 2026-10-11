#pragma once

//Provides common interface for doing C[M, N] = A[M, K] * B[K, N]
//All float pointers refer to device memory holding contiguous row-major floats
//Contents of output c_d will be overwritten; previous values not used
typedef void (*GemmCaller) (int M, int N, int K, const float *a_d, const float *b_d, float *c_d);

//All of the following require GPU with Ampere architecture or newer to do TF32 math on Tensor Cores

//Computes the reference result using cuBLAS with TF32 math enabled
void run_cublas_reference(int M, int N, int K, const float *a_d, const float *b_d, float *c_d);

//Launches the WMMA implementation that loads tiles directly from global memory to registers
void launch_wmma_simple(int M, int N, int K, const float *a_d, const float *b_d, float *c_d);

//Launches the WMMA implementation that stages tiles in shared memory for reuse across warps in a block 
void launch_wmma_shared(int M, int N, int K, const float *a_d, const float *b_d, float *c_d);
 
