#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include "dcis_common.h"


__global__ void peakdet_kernel(const float *input, float *output, const int h, const int w)
{
    int i, j;
    int b_i = blockIdx.x;
    int h_i = blockIdx.y;
    int w_i = threadIdx.x;
    int base = b_i*h*w;
    int max_h_i=-1, max_w_i=-1;
    float max_score=-999, score;
    for (i=h_i-1; i<=h_i+1; i++) {
        if ((i>=0) && (i<h)) {
            for (j=w_i-1; j<=w_i+1; j++) {
                if ((j>=0) && (j<w)) {
                    score = input[base + i*w + j];
                    if (score>max_score) {
                        max_score = score;
                        max_h_i = i;
                        max_w_i = j;
                    }
                }
            }
        }
    }
    if ((max_h_i==h_i) && (max_w_i==w_i)) {
        output[base + h_i*w + w_i] = max_score;
    }
}


at::Tensor peakdet_cuda(const at::Tensor &input)
{
    /*
    input: F(b, h, w)
    w <= 512
    */
    const int b = input.size(0);
    const int h = input.size(1);
    const int w = input.size(2);
    auto output = at::zeros({b, h, w}, input.options());
    if (output.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return output;
    }
    dim3 grid(b, h), block(w);
    peakdet_kernel<<<grid, block>>>(
        input.contiguous().data<float>(),
        output.contiguous().data<float>(),
        h, w);
    THCudaCheck(cudaGetLastError());
    return output;
}
