from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='assign_cuda',
    ext_modules=[
        CUDAExtension(
            'assign_cuda', 
            ['assign_cuda.cpp', 'assign_fcos_kernel.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
})
