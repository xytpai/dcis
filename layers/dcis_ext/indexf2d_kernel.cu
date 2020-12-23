#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include "dcis_common.h"


__global__ void indexf2d_forward_kernel(const int nthreads, 
    const float *input, const float *index, float *output, 
    const int c, const int h, const int w, const int n)
{
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        int w_i = i % w;
        int h_i = (i / w) % h;
        int n_i = i / w / h;
        float fidx = index[i];
        if(fidx<=0 || fidx>=c-1) continue;
        int l_idx = (int)fidx;
        int r_idx = l_idx + 1;
        float l = input[l_idx*h*w+h_i*w+w_i];
        float r = input[r_idx*h*w+h_i*w+w_i];
        float dx = fidx - l_idx;
        output[i] = l*(1.0-dx) + r*dx;
    }
}


at::Tensor indexf2d_forward_cuda(const at::Tensor &input, const at::Tensor &index)
{
    // input: F(c, h, w)
    // index: F(n, h, w)
    // -> F(n, h, w)
    const int c = input.size(0);
    const int h = input.size(1);
    const int w = input.size(2);
    const int n = index.size(0);
    auto output = at::zeros({n, h, w}, input.options());
    if (output.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return output;
    }
    const int nthreads = n*h*w;
    dim3 grid(std::min(DivUp(nthreads, MAX_BLOCK_SIZE), MAX_GRID_SIZE)), block(MAX_BLOCK_SIZE);
    indexf2d_forward_kernel<<<grid, block>>>(nthreads, 
        input.contiguous().data<float>(),
        index.contiguous().data<float>(),
        output.contiguous().data<float>(),
        c, h, w, n);
    THCudaCheck(cudaGetLastError());
    return output;
}


__global__ void indexf2d_backward_kernel(const int nthreads, 
    const float *d_losses, const float *index, float *output, 
    const int c, const int h, const int w, const int n)
{
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        int w_i = i % w;
        int h_i = (i / w) % h;
        int n_i = i / w / h;
        float fidx = index[i];
        if(fidx<=0 || fidx>=c-1) continue;
        int l_idx = (int)fidx;
        int r_idx = l_idx + 1;
        float t = d_losses[i];
        atomicAdd(&output[l_idx*h*w+h_i*w+w_i], (1.0-dx)*t);
        atomicAdd(&output[r_idx*h*w+h_i*w+w_i], dx*t);
    }
}


at::Tensor indexf2d_backward_cuda(const at::Tensor &d_losses, const at::Tensor &index, const int c)
{
    // d_losses: F(n, h, w)
    // index: F(n, h, w)
    // -> F(c, h, w)
    const int n = index.size(0);
    const int h = index.size(1);
    const int w = index.size(2);
    auto output = at::zeros({c, h, w}, index.options());
    if (output.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return output;
    }
    const int nthreads = n*h*w;
    dim3 grid(std::min(DivUp(nthreads, MAX_BLOCK_SIZE), MAX_GRID_SIZE)), block(MAX_BLOCK_SIZE);
    indexf2d_backward_kernel<<<grid, block>>>(nthreads, 
        d_losses.contiguous().data<float>(),
        index.contiguous().data<float>(),
        output.contiguous().data<float>(),
        c, h, w, n);
    THCudaCheck(cudaGetLastError());
    return output;
}