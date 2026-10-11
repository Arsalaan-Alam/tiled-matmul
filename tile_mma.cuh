#pragma once
#include <mma.h>

// Each warp performs [16, 8] * [8, 16] -> [16, 16]
// In our notation: [WMMA_M, WMMA_K] * [WMMA_K, WMMA_N] -> [WMMA_M, WMMA_N]
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 8;

// Each thread holds a part of its warp's float accumulator tile
// Mapping of matrix elements to thread registers is handled by CUDA library
typedef struct  {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
} Accumulator;

__device__ void zero(Accumulator* acc) {
  nvcuda::wmma::fill_fragment(acc->acc_frag, 0.0f);
};

// Accumulates one tile product: acc += A_tile * B_tile
// All 32 threads in a warp cooperatively do this

//Inputs are row-major. lda & ldb are row-strides
// (number of elements to reach same col in next row)

__device__ void tile_mma(Accumulator *acc, const float *a_tile_ptr, int lda, const float *b_tile_ptr, int ldb) {

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_frag;
 
  nvcuda::wmma::load_matrix_sync(a_frag, a_tile_ptr, lda);
  nvcuda::wmma::load_matrix_sync(b_frag, b_tile_ptr, ldb);

  //Typecasting input floats to TF32
  for (int t = 0; t < a_frag.num_elements; t++) {
    a_frag.x[t] = nvcuda::wmma::__float_to_tf32(a_frag.x[t]);
    b_frag.x[t] = nvcuda::wmma::__float_to_tf32(b_frag.x[t]);
  }

  //Output acc retains float precision (no typecasting needed)
  nvcuda::wmma::mma_sync(acc->acc_frag, a_frag, b_frag, acc->acc_frag);
}

// Stores warp's completed 16x16 tile in row-major order; done cooperatively by all 32 threads
// ldc: row-stride of output matrix
__device__ void store_tile(float *c_tile_ptr, int ldc, const Accumulator *acc) {
  nvcuda::wmma::store_matrix_sync(c_tile_ptr, acc->acc_frag, ldc, nvcuda::wmma::mem_row_major); //ldc = N
}


