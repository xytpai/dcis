#include <torch/extension.h>
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")


at::Tensor peakdet_cuda(const at::Tensor &input);
at::Tensor peakdet(const at::Tensor &input)
{
    CHECK_CUDA(input);
    return peakdet_cuda(input);
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
    m.def("assign_centernet", &assign_centernet, "assign_centernet (CUDA)");
}
