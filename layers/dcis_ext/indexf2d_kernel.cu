#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include "dcis_common.h"


at::Tensor indexf2d_forward_cuda(const at::Tensor &input, const at::Tensor &index)
{
    // input: F(c, h, w)
    // index: F(n, h, w)
    // -> F(n, h, w)
    const int c = input.size(0);
    const int h = input.size(1);
    const int w = input.size(2);
    const int n = index.size(0);
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
