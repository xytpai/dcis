#include <torch/extension.h>
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")


at::Tensor peakdet_cuda(const at::Tensor &input);
at::Tensor peakdet(const at::Tensor &input)
{
    CHECK_CUDA(input);
    return peakdet_cuda(input);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("peakdet", &peakdet, "peakdet (CUDA)");
}