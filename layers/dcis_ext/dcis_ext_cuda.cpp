#include <torch/extension.h>
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")


at::Tensor peakdet_cuda(const at::Tensor &input);
at::Tensor peakdet(const at::Tensor &input)
{
    CHECK_CUDA(input);
    return peakdet_cuda(input);
}


at::Tensor indexf2d_forward_cuda(const at::Tensor &input, const at::Tensor &index);
at::Tensor indexf2d_forward(const at::Tensor &input, const at::Tensor &index)
{
    CHECK_CUDA(input);
    CHECK_CUDA(index);
    return indexf2d_forward_cuda(input, index);
}


at::Tensor indexf2d_backward_cuda(const at::Tensor &d_losses, const at::Tensor &index, const int c);
at::Tensor indexf2d_backward(const at::Tensor &d_losses, const at::Tensor &index, const int c)
{
    CHECK_CUDA(d_losses);
    CHECK_CUDA(index);
    return indexf2d_backward_cuda(d_losses, index, c);
}


at::Tensor assign_centernet_cuda(
    const at::Tensor &cls_idx,
    const at::Tensor &bbox,
    const int stride, const int num_class);
at::Tensor assign_centernet(
    const at::Tensor &cls_idx,
    const at::Tensor &bbox,
    const int stride, const int num_class)
{
    CHECK_CUDA(cls_idx);
    CHECK_CUDA(bbox);
    return assign_centernet_cuda(cls_idx, bbox, stride, num_class);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("peakdet", &peakdet, "peakdet (CUDA)");
    m.def("indexf2d_forward", &indexf2d_forward, "indexf2d_forward (CUDA)");
    m.def("indexf2d_backward", &indexf2d_backward, "indexf2d_backward (CUDA)");
    m.def("assign_centernet", &assign_centernet, "assign_centernet (CUDA)");
}
