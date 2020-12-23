from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dcis_ext_cuda',
    ext_modules=[
        CUDAExtension(
            'dcis_ext_cuda', ['dcis_ext_cuda.cpp',
            'peakdet_kernel.cu', 'indexf2d_kernel.cu',
            'indexf2d_kernel.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
})