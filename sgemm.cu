#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <assert.h>
#include <cmath>
#include "gemm.h"
#include "utils.h"

// Matrix dimensions notation: C[M, N] = A[M, K] * B[K, N]
#define MATRIX_M 4096
#define MATRIX_N 4096
#define MATRIX_K 4096

struct KernelEntry {
  const char* name;
  GemmCaller caller;
};

// Implementations to validate and optionally benchmark
KernelEntry kernels[] = {
  {"Simple warp MMA", launch_wmma_simple},
  {"Shared memory warp MMA", launch_wmma_shared},
};

int main(int argc, char *argv[]) {

  bool benchmark = false;

  // Parse arguments
  if (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0) {
    benchmark = true;
  } else if (argc != 1) {
    fprintf(stderr, "Usage: %s [--benchmark]\n", argv[0]);
    exit(1);
  }

  // Allocate and initialize matrices
  // Our convention: all matrices are stored in row-major order

  // Host matrices
  float *a_h, *b_h, *c_h, *c_h_ref;
  a_h = (float *) malloc(MATRIX_M * MATRIX_K * sizeof(float));
  b_h = (float *) malloc(MATRIX_K * MATRIX_N * sizeof(float));
  c_h = (float *) malloc(MATRIX_M * MATRIX_N * sizeof(float));
  c_h_ref = (float *) malloc(MATRIX_M * MATRIX_N * sizeof(float));

  // Device matrices
  float *a_d, *b_d, *c_d, *c_d_ref;
  CUDA_CHECK(cudaMalloc((void **)&a_d, MATRIX_M * MATRIX_K * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&b_d, MATRIX_K * MATRIX_N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&c_d, MATRIX_M * MATRIX_N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&c_d_ref, MATRIX_M * MATRIX_N * sizeof(float)));

  srand(4103);

  fill_random(a_h, MATRIX_M * MATRIX_K);
  fill_random(b_h, MATRIX_K * MATRIX_N);

  printf("\nM = %d, N = %d, K = %d\n", MATRIX_M, MATRIX_N, MATRIX_K);

  // Transfer inputs to device
  CUDA_CHECK(cudaMemcpy(a_d, a_h, MATRIX_M * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_d, b_h, MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
 
  // Run reference cuBLAS kernel
  run_cublas_reference(MATRIX_M, MATRIX_N, MATRIX_K, a_d, b_d, c_d_ref);
  CUDA_CHECK(cudaMemcpy(c_h_ref, c_d_ref, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

  // Validate each wmma kernel
  for (auto kernel : kernels) {
    // Filling outputs with NaN so correctness check catches elements kernel fails to write 
    CUDA_CHECK(cudaMemset(c_d, 0xFF, MATRIX_M * MATRIX_N * sizeof(float)));
    kernel.caller(MATRIX_M, MATRIX_N, MATRIX_K, a_d, b_d, c_d);
    CUDA_CHECK(cudaMemcpy(c_h, c_d, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    compare_results(c_h, c_h_ref, MATRIX_M * MATRIX_N, kernel.name);
  }
  
  printf("Test comparison with cuBLAS passed for both wmma_simple and wmma_shared\n");

  // Optionally benchmark each wmma kernel
  if (benchmark) {
    for (auto kernel : kernels) {
      float measured_ms = benchmark_ms(kernel.caller, 
                      MATRIX_M, MATRIX_N, MATRIX_K, a_d, b_d, c_d);
      printf("%s time (ms) = %9.3f\n", kernel.name, measured_ms);
    }
  }

  // Release resources
  CUDA_CHECK(cudaFree(a_d));
  CUDA_CHECK(cudaFree(b_d));
  CUDA_CHECK(cudaFree(c_d_ref));
  CUDA_CHECK(cudaFree(c_d));

  free(a_h);
  free(b_h);
  free(c_h_ref);
  free(c_h);
 
  return 0;
}
