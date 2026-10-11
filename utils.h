#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "gemm.h"

//Exits of failure; reports error name, file and line number at calling point
#define CUDA_CHECK(call) do { \
  cudaError_t _cuda_err = (call); \
  if (_cuda_err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_cuda_err)); \
    exit(1); \
  } \
} while (0)

void fill_random(float *arr, int N);

void compare_results(const float *a, const float *b, int N, const char *name);

float benchmark_ms(GemmCaller caller, int M, int N, int K, const float *a_d, const float *b_d, float *c_d);



