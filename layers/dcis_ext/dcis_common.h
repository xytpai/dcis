#ifndef DCIS_COMMON_H
#define DCIS_COMMON_H


#define MAX_BLOCK_SIZE 512
#define MAX_GRID_SIZE 2048
#define DivUp(x,y) (int)ceil((float)x/y)
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
           i += blockDim.x * gridDim.x)


#endif