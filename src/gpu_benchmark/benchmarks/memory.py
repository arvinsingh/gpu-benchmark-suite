"""
Memory-related benchmarks comparing CUDA, Triton, and PyTorch implementations.
"""

import torch
import triton
import triton.language as tl
import numpy as np
from typing import Tuple

try:
    import cuda_kernels
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


# =============================================================================
# Vector Addition Benchmarks
# =============================================================================

def vector_add_pytorch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch vector addition implementation."""
    return a + b


@triton.jit
def vector_add_triton_kernel(
    x_ptr,  # ptr to first i/p vector
    y_ptr,  # ptr to second input vector
    output_ptr,  # ptr to output vector
    n_elements,  # size of the vector
    BLOCK_SIZE: tl.constexpr,  # block size
):
    """Triton kernel for vector addition."""
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton vector addition implementation."""
    n_elements = a.numel()
    output = torch.empty_like(a)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vector_add_triton_kernel[grid](
        a, b, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def vector_add_cuda(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CUDA vector addition implementation."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    return cuda_kernels.vector_add(a, b)


# =============================================================================
# Vector Multiplication Benchmarks  
# =============================================================================

def vector_mul_pytorch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch vector multiplication implementation."""
    return a * b


@triton.jit
def vector_mul_triton_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for element-wise vector multiplication."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_mul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton vector multiplication implementation."""
    n_elements = a.numel()
    output = torch.empty_like(a)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vector_mul_triton_kernel[grid](
        a, b, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def vector_mul_cuda(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CUDA vector multiplication implementation."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    return cuda_kernels.vector_mul(a, b)


# =============================================================================
# Memory Bandwidth Benchmarks
# =============================================================================

def memory_copy_pytorch(a: torch.Tensor) -> torch.Tensor:
    """PyTorch memory copy (bandwidth test)."""
    return a.clone()


@triton.jit
def memory_copy_triton_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for memory copy."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)


def memory_copy_triton(a: torch.Tensor) -> torch.Tensor:
    """Triton memory copy implementation."""
    n_elements = a.numel()
    output = torch.empty_like(a)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    memory_copy_triton_kernel[grid](
        a, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def memory_copy_cuda(a: torch.Tensor) -> torch.Tensor:
    """CUDA memory copy implementation."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    return cuda_kernels.memory_copy(a)


# =============================================================================
# Memory Access Pattern Benchmarks
# =============================================================================

def strided_access_pytorch(a: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """PyTorch strided memory access."""
    return a[::stride]


@triton.jit  
def strided_access_triton_kernel(
    input_ptr, output_ptr, n_elements, stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for strided memory access."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_offsets = offsets * stride
    input_mask = input_offsets < (n_elements * stride)
    
    data = tl.load(input_ptr + input_offsets, mask=input_mask & mask)
    tl.store(output_ptr + offsets, data, mask=mask)


def strided_access_triton(a: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """Triton strided memory access implementation."""
    n_elements = (a.numel() + stride - 1) // stride  # Output size
    output = torch.empty(n_elements, device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    strided_access_triton_kernel[grid](
        a, output, n_elements, stride, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def strided_access_cuda(a: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """CUDA strided memory access implementation."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    return cuda_kernels.strided_access(a, stride)


# =============================================================================
# Performance Counter Functions
# =============================================================================

def vector_flops_counter(a: torch.Tensor, b: torch.Tensor) -> float:
    """Count FLOPS for vector operations (one operation per element)."""
    return float(a.numel())


def vector_bandwidth_counter(a: torch.Tensor, b: torch.Tensor) -> float:
    """Count memory bandwidth for vector operations (read 2 vectors, write 1)."""
    element_size = a.element_size()  # bytes per element
    total_elements = a.numel() * 3  # 2 inputs + 1 output
    return float(total_elements * element_size)


def memory_copy_bandwidth_counter(a: torch.Tensor) -> float:
    """Count memory bandwidth for memory copy (read 1, write 1)."""
    element_size = a.element_size()
    total_elements = a.numel() * 2  # 1 input + 1 output
    return float(total_elements * element_size)


def strided_bandwidth_counter(a: torch.Tensor, stride: int = 2) -> float:
    """Count memory bandwidth for strided access."""
    element_size = a.element_size()
    output_elements = (a.numel() + stride - 1) // stride
    # Reading strided data + writing output
    total_bytes = (output_elements * stride + output_elements) * element_size
    return float(total_bytes)


# =============================================================================
# Benchmark Registration
# =============================================================================

def register_memory_benchmarks(runner):
    """Register all memory benchmarks with the runner."""
    
    # Vector addition
    runner.register_benchmark(
        "memory", "vector-add",
        cuda_impl=vector_add_cuda if CUDA_AVAILABLE else None,
        triton_impl=vector_add_triton,
        pytorch_impl=vector_add_pytorch,
        flops_counter=vector_flops_counter,
        bandwidth_counter=vector_bandwidth_counter,
    )
    
    # Vector multiplication  
    runner.register_benchmark(
        "memory", "vector-mul",
        cuda_impl=vector_mul_cuda if CUDA_AVAILABLE else None,
        triton_impl=vector_mul_triton,
        pytorch_impl=vector_mul_pytorch,
        flops_counter=vector_flops_counter,
        bandwidth_counter=vector_bandwidth_counter,
    )
    
    # Memory copy
    runner.register_benchmark(
        "memory", "memory-copy",
        cuda_impl=memory_copy_cuda if CUDA_AVAILABLE else None,
        triton_impl=memory_copy_triton,
        pytorch_impl=memory_copy_pytorch,
        bandwidth_counter=memory_copy_bandwidth_counter,
    )
    
    # Strided access
    runner.register_benchmark(
        "memory", "strided-access",
        cuda_impl=strided_access_cuda if CUDA_AVAILABLE else None,
        triton_impl=strided_access_triton,
        pytorch_impl=strided_access_pytorch,
        bandwidth_counter=strided_bandwidth_counter,
    )
