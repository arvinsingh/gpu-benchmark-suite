"""
Mathematical operation benchmarks comparing CUDA, Triton, and PyTorch implementations.
"""

import math
from typing import Tuple

import numpy as np
import torch
import triton
import triton.language as tl

try:
    import cuda_kernels

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


# =============================================================================
# Matrix Multiplication Benchmarks
# =============================================================================


def matmul_pytorch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch matrix multiplication implementation."""
    return torch.matmul(A, B)


@triton.jit
def matmul_triton_kernel(
    a_ptr,
    b_ptr,
    c_ptr,  # ptrs to matrices
    M,
    N,
    K,  # matrix dimensions
    stride_am,
    stride_ak,  # strides
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton matrix multiplication kernel."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # load the next block of A and B, generate a mask by checking K dimension
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # accumulate along K dimension
        accumulator += tl.dot(a, b)

        # advance ptrs to next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Triton matrix multiplication implementation."""

    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    M, K = A.shape
    K, N = B.shape

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    matmul_triton_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return C


def matmul_cuda_naive(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CUDA naive matrix multiplication implementation."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    return cuda_kernels.matmul_naive(A, B)


def matmul_cuda_shared(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CUDA shared memory matrix multiplication implementation."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    return cuda_kernels.matmul_shared(A, B)


# =============================================================================
# Reduction Operations
# =============================================================================


def reduce_sum_pytorch(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reduction sum implementation."""
    return torch.sum(x)


@triton.jit
def reduce_sum_triton_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton reduction sum kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    result = tl.sum(x)

    tl.store(output_ptr + pid, result)


def reduce_sum_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton reduction sum implementation."""
    n_elements = x.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    temp = torch.empty(grid[0], device=x.device, dtype=x.dtype)

    reduce_sum_triton_kernel[grid](x, temp, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # multiple blocks, reduce again
    if grid[0] > 1:
        return reduce_sum_triton(temp)
    else:
        return temp


def reduce_sum_cuda(x: torch.Tensor) -> torch.Tensor:
    """CUDA reduction sum implementation."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    return cuda_kernels.reduce_sum(x)


def reduce_sum_cuda_shared(x: torch.Tensor) -> torch.Tensor:
    """CUDA shared memory reduction sum implementation."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    return cuda_kernels.reduce_sum_shared(x)


# =============================================================================
# Activation Functions
# =============================================================================


def relu_pytorch(x: torch.Tensor) -> torch.Tensor:
    """PyTorch ReLU implementation."""
    return torch.relu(x)


@triton.jit
def relu_triton_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton ReLU kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    result = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, result, mask=mask)


def relu_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton ReLU implementation."""
    output = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    relu_triton_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output


def gelu_pytorch(x: torch.Tensor) -> torch.Tensor:
    """PyTorch GELU implementation."""
    return torch.nn.functional.gelu(x)


@triton.jit
def gelu_triton_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton GELU kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x * x * x
    tanh_input = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    tanh_result = tl.libdevice.tanh(tanh_input)
    result = 0.5 * x * (1.0 + tanh_result)

    tl.store(output_ptr + offsets, result, mask=mask)


def gelu_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton GELU implementation."""
    output = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    gelu_triton_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output


# =============================================================================
# Softmax Implementation
# =============================================================================


def softmax_pytorch(x: torch.Tensor) -> torch.Tensor:
    """PyTorch softmax implementation."""
    return torch.softmax(x, dim=-1)


@triton.jit
def softmax_triton_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton softmax kernel."""
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    # load row data
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # compute softmax
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # store result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton softmax implementation."""
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    output = torch.empty_like(x)

    softmax_triton_kernel[(n_rows,)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# =============================================================================
# Performance Counter Functions
# =============================================================================


def matmul_flops_counter(A: torch.Tensor, B: torch.Tensor) -> float:
    """Count FLOPS for matrix multiplication (2*M*N*K operations)."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    return float(2 * M * N * K)


def matmul_bandwidth_counter(A: torch.Tensor, B: torch.Tensor) -> float:
    """Count memory bandwidth for matrix multiplication."""
    element_size = A.element_size()
    total_elements = A.numel() + B.numel() + (A.shape[0] * B.shape[1])  # A + B + C
    return float(total_elements * element_size)


def reduce_flops_counter(x: torch.Tensor) -> float:
    """Count FLOPS for reduction (N-1 additions)."""
    return float(x.numel() - 1)


def activation_flops_counter(x: torch.Tensor) -> float:
    """Count FLOPS for activation functions (1 per element)."""
    return float(x.numel())


def activation_bandwidth_counter(x: torch.Tensor) -> float:
    """Count memory bandwidth for activation functions (read input, write output)."""
    element_size = x.element_size()
    total_elements = x.numel() * 2  # input + output
    return float(total_elements * element_size)


# =============================================================================
# Benchmark Registration
# =============================================================================


def register_math_benchmarks(runner):
    """Register all math benchmarks with the runner."""

    # Matrix multiplication - naive
    runner.register_benchmark(
        "math",
        "matmul-naive",
        cuda_impl=matmul_cuda_naive if CUDA_AVAILABLE else None,
        triton_impl=matmul_triton,
        pytorch_impl=matmul_pytorch,
        flops_counter=matmul_flops_counter,
        bandwidth_counter=matmul_bandwidth_counter,
    )

    # Matrix multiplication - optimized
    runner.register_benchmark(
        "math",
        "matmul-shared",
        cuda_impl=matmul_cuda_shared if CUDA_AVAILABLE else None,
        triton_impl=matmul_triton,
        pytorch_impl=matmul_pytorch,
        flops_counter=matmul_flops_counter,
        bandwidth_counter=matmul_bandwidth_counter,
    )

    # Reduction sum
    runner.register_benchmark(
        "math",
        "reduce-sum",
        cuda_impl=reduce_sum_cuda if CUDA_AVAILABLE else None,
        triton_impl=reduce_sum_triton,
        pytorch_impl=reduce_sum_pytorch,
        flops_counter=reduce_flops_counter,
    )

    # Reduction sum - shared memory
    runner.register_benchmark(
        "math",
        "reduce-sum-shared",
        cuda_impl=reduce_sum_cuda_shared if CUDA_AVAILABLE else None,
        triton_impl=reduce_sum_triton,
        pytorch_impl=reduce_sum_pytorch,
        flops_counter=reduce_flops_counter,
    )

    # ReLU activation
    runner.register_benchmark(
        "math",
        "relu",
        triton_impl=relu_triton,
        pytorch_impl=relu_pytorch,
        flops_counter=activation_flops_counter,
        bandwidth_counter=activation_bandwidth_counter,
    )

    # GELU activation
    runner.register_benchmark(
        "math",
        "gelu",
        triton_impl=gelu_triton,
        pytorch_impl=gelu_pytorch,
        flops_counter=activation_flops_counter,
        bandwidth_counter=activation_bandwidth_counter,
    )

    # Softmax
    runner.register_benchmark(
        "math",
        "softmax",
        triton_impl=softmax_triton,
        pytorch_impl=softmax_pytorch,
        flops_counter=activation_flops_counter,
        bandwidth_counter=activation_bandwidth_counter,
    )
