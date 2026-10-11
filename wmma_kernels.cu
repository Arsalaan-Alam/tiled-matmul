#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>
#include <assert.h>
#include "tile_mma.cuh"
#include "gemm.h"
#include "utils.h"

using namespace nvcuda;

// Each block contains warps arranged as a grid of output tiles
// For our (64, 2) block dimensions, each block has 2 x 2 grid of warps
// Each warp computes 16 x 16 output tile
__global__ void wmma_simple(const float *a, const float *b, float *c, int M, int N, int K) {
  //TODO: complete this kernel
}

void launch_wmma_simple(int M, int N, int K, const float *a_d, const float *b_d, float *c_d) {

    assert(M % WMMA_M == 0 && N % WMMA_N == 0 && K % WMMA_K == 0);
    dim3 gridDim;
    dim3 blockDim;

    // 128 threads -> four warps in a block in 2 x 2 arrangement
    // One block computes 32 x 32 tile of output
    // Notice how this is different from the scalar matmul case (1 thread -> 1 output element)
    blockDim.x = 64;
    blockDim.y = 2;

    //Tiling the output matrix
    gridDim.x = N / (WMMA_N * blockDim.x / 32); //warpSize builtin not available on Host
    gridDim.y = M / (WMMA_M * blockDim.y);
    
    wmma_simple <<< gridDim, blockDim >>> (a_d, b_d, c_d, M, N, K);

    CUDA_CHECK(cudaGetLastError()); // checking launch error
}


__global__ void wmma_shared(const float *a, const float *b, float *c, int M, int N, int K) {

  Accumulator acc;
  zero(&acc);
 
  //Align the start address to 4 bytes to hold float data
  __shared__ __align__(32) float a_tile[2 * WMMA_M][WMMA_K];
  __shared__ __align__(32) float b_tile[WMMA_K][2 * WMMA_N];

  //Flattening threads coordinates by tid to distribute tile loads across the block
  //Global to shared memory tile loadings explictly done using scalar loads by threads
  //(rather than warp-level loading)
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int num_threads = blockDim.x * blockDim.y; 

  //Element coordinates of the block's top-left output corner
  int block_row = blockIdx.y * 2 * WMMA_M;
  int block_col = blockIdx.x * 2 * WMMA_N;

  //This warp's position in the block's 2 x 2 warp grid
  int warp_row = threadIdx.y;
  int warp_col = threadIdx.x / warpSize;

  //Staging 32 x 8 sized A_tile and 8 by 32 sized B_tile
  //Cooperatively loaded by all threads in the block 
  for (int k = 0; k < K; k += WMMA_K) {
      //Each A half is reused by two warp columns
      for (int idx = tid; idx < 2 * WMMA_M * WMMA_K; idx += num_threads) {
        int row = idx / WMMA_K;
        int col = idx % WMMA_K;

        // TODO: Fill global memory source address for matrix a
        // a_tile[row][col] = a[?]; 
      }

      //Each B half is reused by two warp rows
      for (int idx = tid; idx < WMMA_K * 2 * WMMA_N; idx += num_threads) {
        int row = idx / (2 * WMMA_N);
        int col = idx % (2 * WMMA_N);

        // TODO: Fill global memory source address for matrix b
        // b_tile[row][col] = b[?]; 
      }
 
      __syncthreads();
      // TODO: Supply arguments to tile_mma()
      // Hint: Shared-memory tile pointers and leading dimensions
      // tile_mma(?);
      __syncthreads();
  }

  int c_row = block_row + warp_row * WMMA_M;
  int c_col = block_col + warp_col * WMMA_N;
  float *c_tile_ptr = c + c_row * N + c_col;
  //Storing this warp's 16 x 16 output to global memory
  store_tile(c_tile_ptr, N, &acc);
}

void launch_wmma_shared(int M, int N, int K, const float *a_d, const float *b_d, float *c_d) {

    assert(M % 32 == 0 && N % 32 == 0 && K % 8 == 0);

    dim3 gridDim;
    dim3 blockDim;

    blockDim.x = 64;
    blockDim.y = 2;

    gridDim.x = N / (WMMA_N * blockDim.x / 32); //warpSize builtin not available on Host
    gridDim.y = M / (WMMA_M * blockDim.y);
    
    wmma_shared <<< gridDim, blockDim >>> (a_d, b_d, c_d, M, N, K);

    CUDA_CHECK(cudaGetLastError()); // checking launch error

}


