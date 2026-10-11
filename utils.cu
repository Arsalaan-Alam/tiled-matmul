#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "gemm.h"
#include "utils.h"

void fill_random(float *arr, int N) {
  //Fill host matrix with random values in [-1, 1]
  for (int i = 0; i < N; i++) 
      arr[i] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

void compare_results(const float *a, const float *b, int N, const char *name) {

  float delta = 0.0;

  for (int i = 0; i < N; i++) {
    delta = fabs((float) a[i] - (float) b[i]);
    //1e-2 threshold is arbitrary here
    if (isnan(a[i]) || isnan(b[i]) || delta > 1e-2) { // report failure if a NaN encountered
      fprintf(stderr, "%s: error. %f, %f, %d", name, a[i], b[i], i);
      fprintf(stderr, "Failed to match reference.\n");
      exit(1);
    }
  }
}

float benchmark_ms(GemmCaller caller, int M, int N, int K, const float *a_d, const float *b_d, float *c_d) {

  //untimed warmups
  for (int i = 0; i < 5; i++) {
    caller(M, N, K, a_d, b_d, c_d);
  }
 
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  //Timing averaged over 100 runs
  for (int i = 0; i < 100; i++) {
    caller(M, N, K, a_d, b_d, c_d);
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
 
  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return total_ms / 100;
}


