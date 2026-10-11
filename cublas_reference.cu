#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gemm.h" 

//Exits of failure; reports error name, file and line number at calling point
//do { } while (0) provides a convenient way to write multi-line macro
//Having a macro instead of a function call lets us intercept the file and line no.
#define CUBLAS_CHECK(call) do { \
  cublasStatus_t _cublas_err = (call); \
  if (_cublas_err != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cublas error %s:%d: %s\n", __FILE__, __LINE__, cublasGetStatusString(_cublas_err)); \
    exit(1); \
  } \
} while (0)

void run_cublas_reference(int M, int N, int K, const float *a_d, const float *b_d, float *c_d) {
    cublasHandle_t handle = NULL;
    
    CUBLAS_CHECK(cublasCreate(&handle));

    //Allowing cuBLAS to use TF32 Tensor Core ops
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    printf("Running matmul with cuBLAS\n");   

    //GEMM computes C = alpha * A * B + beta * C
    //These values compute A * B without reading previous contents of C
    float alpha = 1.0, beta = 0.0;

    // cuBLAS expects matrices to be column-major stored. The same bytes representing
    // row-major A[M, K] represent colum-major A^T[K, M] (same for matrices B and C)

    // We compute C^T[N, M] = B^T[N, K] * A^T[K, M] by swapping the operands
    // Colum-major result has memory layout as row-major C[M, N]; no explicit transpose needed

    // Leading dimension -> how many elements to we need to stride to reach next column at same row 
    // (for column-major view); K for A^T, N for B^T, N for C^T
    CUBLAS_CHECK( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b_d, N, a_d, K, &beta, c_d, N));
    CUBLAS_CHECK(cublasDestroy(handle));
}
