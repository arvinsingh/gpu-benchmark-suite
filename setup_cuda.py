#!/usr/bin/env python3
"""
CUDA setup script.
"""

import os
import sys
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch

def setup_cuda_extensions():
    """Set up CUDA extensions with proper compiler settings."""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, cannot build CUDA extensions")
        sys.exit(1)
    
    print("✅ CUDA detected, proceeding with build...")
    
    # default GCC and try to work around compatibility issues
    os.environ.pop("CC", None)
    os.environ.pop("CXX", None)
    
    # default system paths
    cuda_home = "/usr/lib/nvidia-cuda-toolkit"
    os.environ["CUDA_HOME"] = cuda_home
    
    project_root = Path(__file__).parent
    kernels_dir = project_root / "kernels"
    
    print(f"Kernels directory: {kernels_dir}")

    # verify source files exist
    cpp_file = kernels_dir / "cuda_bindings.cpp"
    cu_file = kernels_dir / "vector_kernels.cu"
    header_file = kernels_dir / "vector_kernels.cuh"

    if not cpp_file.exists():
        print(f"❌ C++ binding file not found: {cpp_file}")
        sys.exit(1)
    if not cu_file.exists():
        print(f"❌ CUDA source file not found: {cu_file}")
        sys.exit(1)
    if not header_file.exists():
        print(f"❌ CUDA header file not found: {header_file}")
        sys.exit(1)
    
    print("✅ All source files found")
    
    # Get CUDA compute capability
    try:
        device = torch.cuda.get_device_properties(0)
        compute_capability = f"{device.major}{device.minor}"
        arch_flags = [f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}"]
        print(f"Target GPU compute capability: {compute_capability}")
    except:
        # fallback to common architectures
        arch_flags = [
            "-gencode=arch=compute_75,code=sm_75",  # RTX 20xx
            "-gencode=arch=compute_80,code=sm_80",  # A100
            "-gencode=arch=compute_86,code=sm_86",  # RTX 30xx
        ]
        print("Using fallback GPU architectures")
    
    # define the CUDA extension
    cuda_extension = CUDAExtension(
        name="cuda_kernels",
        sources=[
            str(cpp_file),
            str(cu_file),
        ],
        include_dirs=[
            str(kernels_dir),
        ],
        extra_compile_args={
            'cxx': [
                '-O3', 
                '-DWITH_CUDA',
                '-fPIC',
            ],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-DWITH_CUDA',
                '--expt-relaxed-constexpr',
            ] + arch_flags,
        },
        verbose=True,
    )

    # build the extension
    setup(
        name="cuda_kernels",
        ext_modules=[cuda_extension],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
    )

if __name__ == "__main__":
    setup_cuda_extensions()
