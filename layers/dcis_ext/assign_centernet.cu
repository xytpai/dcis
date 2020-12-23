#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include "dcis_common.h"


__device__ float gaussian_radius(float height, float width)
{
    const float min_overlap = 0.7;
    float a1  = 1;
    float b1  = height + width;
    float c1  = width * height * (1 - min_overlap) / (1 + min_overlap);
    float sq1 = sqrt(b1 ** 2 - 4 * a1 * c1);
    float r1  = (b1 + sq1) / 2;

    float a2  = 4;
    float b2  = 2 * (height + width);
    float c2  = (1 - min_overlap) * width * height;
    float sq2 = sqrt(b2 ** 2 - 4 * a2 * c2);
    float r2  = (b2 + sq2) / 2;

    float a3  = 4 * min_overlap;
    float b3  = -2 * min_overlap * (height + width);
    float c3  = (min_overlap - 1) * width * height;
    float sq3 = sqrt(b3 ** 2 - 4 * a3 * c3);
    float r3  = (b3 + sq3) / 2;

    return min(r1, min(r2, r3));
}


__global__ void assign_centernet_kernel(const int nthreads,
    const long *cls_idx, const float *bbox, float *output,
    const int ph, const int pw, const int n, const int num_class,
    const int stride)
{
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        int pw_i = i % pw;
        int ph_i = (i / pw) % ph;
        int class_i = i / pw / ph;
        float center_y = ph_i * stride;
        float center_x = pw_i * stride;
        int base_1 = (ph_i*pw+pw_i)*n;
        int base_4 = (ph_i*pw+pw_i)*n*4;
        float res = 0;
        for (int n_i=0; n_i<n; ++n_i) {
            int cls = cls_idx[base_1];
            base_1++;
            if (cls!=class_i) {
                base_4 += 4;
                continue;
            }
            float ymin = bbox[base_4+0];
            float xmin = bbox[base_4+1];
            float ymax = bbox[base_4+2];
            float xmax = bbox[base_4+3];
            float cy = (ymin + ymax)/2.0;
            float cx = (xmin + xmax)/2.0;
            base_4 += 4;
            float oy = cy - center_y;
            float ox = cx - center_x;
            float bh = ymax - ymin + 1;
            float bw = xmax - xmin + 1;
            float radius = gaussian_radius(bh, bw);
            float sigma = (2.0*radius+1)/6.0;
            res = max(res, exp(-(oy*oy+ox*ox)/(2*sigma*sigma)));
        }
        output[i] = res;
    }
}


at::Tensor assign_centernet_cuda(
    const at::Tensor &cls_idx,
    const at::Tensor &bbox,
    const int stride, const int num_class)
{
    // cls_idx: L(ph, pw, n) 0~
    // bbox: F(ph, pw, n, 4) y, x
    // -> F(num_class, ph, pw)
    const int ph = cls_idx.size(0);
    const int pw = cls_idx.size(1);
    const int n  = cls_idx.size(2);
    auto output = at::empty({num_class, ph, pw}, bbox.options());
    if (output.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return output;
    }
    const int nthreads = num_class*ph*pw;
    dim3 grid(std::min(DivUp(nthreads, MAX_BLOCK_SIZE), MAX_GRID_SIZE)), block(MAX_BLOCK_$
    assign_centernet_kernel<<<grid, block>>>(nthreads,
            cls_idx.contiguous().data<long>(),
            bbox.contiguous().data<float>(),
            output.contiguous().data<float>(),
            ph, pw, n, num_class, stride);
    THCudaCheck(cudaGetLastError());
    return output;
}
