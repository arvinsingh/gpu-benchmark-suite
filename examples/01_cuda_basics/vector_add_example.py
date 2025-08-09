"""
CUDA Basics Example: Vector Addition

This example demonstrates the fundamental concepts of CUDA programming:
- Kernel functions
- Memory management
- Thread organization
- Error checking

Learning objectives:
1. Understand the difference between host (CPU) and device (GPU) code
2. Learn how to allocate and manage GPU memory
3. Understand thread indexing and grid/block organization
4. Learn proper error checking practices
"""

import time
from typing import Tuple

import numpy as np
import torch

try:
    import cuda_kernels

    CUDA_AVAILABLE = True
    print("  CUDA kernels available")
except ImportError:
    CUDA_AVAILABLE = False
    print("  CUDA kernels not available - using PyTorch implementations")


def demonstrate_cuda_basics():
    """Demonstrate basic CUDA concepts with vector addition."""
    print("=" * 60)
    print("CUDA BASICS: Vector Addition Example")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("-- CUDA is not available on this system --")
        return

    device = torch.device("cuda:0")
    print(f"   Using device: {device}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Problem size
    N = 1_000_000
    print(f"\n Problem size: {N:,} elements ({N * 4 / 1e6:.1f} MB per vector)")

    # create input data on CPU
    print("\n  Creating input data...")
    a_cpu = torch.randn(N, dtype=torch.float32)
    b_cpu = torch.randn(N, dtype=torch.float32)

    # cpu reference computation
    print("  Computing reference result on CPU...")
    start_time = time.perf_counter()
    c_cpu_ref = a_cpu + b_cpu
    cpu_time = (time.perf_counter() - start_time) * 1000
    print(f"   CPU time: {cpu_time:.2f} ms")

    # Transfer data to gpu
    print("\n  Transferring data to GPU...")
    start_time = time.perf_counter()
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)
    transfer_time = (time.perf_counter() - start_time) * 1000
    print(f"   Transfer time: {transfer_time:.2f} ms")

    # PyTorch GPU computation
    print("\n  Computing with PyTorch on GPU...")
    # warmup
    for _ in range(3):
        _ = a_gpu + b_gpu

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    c_pytorch = a_gpu + b_gpu
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start_time) * 1000
    print(f"   PyTorch GPU time: {pytorch_time:.2f} ms")

    # cuda kernel computation (if available)
    if CUDA_AVAILABLE:
        print("\n  Computing with custom CUDA kernel...")
        # warmup
        for _ in range(3):
            _ = cuda_kernels.vector_add(a_gpu, b_gpu)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        c_cuda = cuda_kernels.vector_add(a_gpu, b_gpu)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start_time) * 1000
        print(f"   CUDA kernel time: {cuda_time:.2f} ms")

        # verify correctness
        cuda_error = torch.max(torch.abs(c_cuda - c_pytorch)).item()
        print(f"   Max error vs PyTorch: {cuda_error:.2e}")

    # transfer result back to CPU
    print("\n  Transferring result back to CPU...")
    start_time = time.perf_counter()
    c_gpu_cpu = c_pytorch.cpu()
    transfer_back_time = (time.perf_counter() - start_time) * 1000
    print(f"   Transfer back time: {transfer_back_time:.2f} ms")

    # verify correctness
    max_error = torch.max(torch.abs(c_gpu_cpu - c_cpu_ref)).item()
    print(f"\n  Verification: Max error = {max_error:.2e}")

    # Performance summary
    print("\n  Performance Summary:")
    print(f"   CPU:           {cpu_time:8.2f} ms")
    print(
        f"   PyTorch GPU:   {pytorch_time:8.2f} ms (speedup: {cpu_time/pytorch_time:.1f}x)"
    )
    if CUDA_AVAILABLE:
        print(
            f"   CUDA kernel:   {cuda_time:8.2f} ms (speedup: {cpu_time/cuda_time:.1f}x)"
        )
    print(f"   Transfer to:   {transfer_time:8.2f} ms")
    print(f"   Transfer from: {transfer_back_time:8.2f} ms")

    # Memory usage
    print(f"\n  GPU Memory Usage:")
    print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
    print(f"   Cached:    {torch.cuda.memory_reserved(0) / 1e6:.1f} MB")


def demonstrate_thread_organization():
    """Demonstrate how CUDA threads are organized."""
    print("\n" + "=" * 60)
    print("THREAD ORGANIZATION")
    print("=" * 60)

    print("   CUDA Thread Hierarchy:")
    print("   Grid → Blocks → Threads")
    print("   - Grid: Collection of blocks")
    print("   - Block: Collection of threads (up to 1024 per block)")
    print("   - Thread: Individual execution unit")

    # Example thread organization for different problem sizes
    problem_sizes = [1024, 4096, 1_000_000]
    block_size = 256  # Common block size

    print(f"\n   Thread Organization Examples (block size = {block_size}):")
    for N in problem_sizes:
        num_blocks = (N + block_size - 1) // block_size  # ceil
        total_threads = num_blocks * block_size
        efficiency = N / total_threads

        print(
            f"   N = {N:>8,}: {num_blocks:>4} blocks × {block_size} threads = {total_threads:>8,} total threads"
        )
        print(
            f"                Efficiency: {efficiency:.1%} (active threads / total threads)"
        )


def demonstrate_memory_concepts():
    """Demonstrate GPU memory concepts."""
    print("\n" + "=" * 60)
    print("MEMORY CONCEPTS")
    print("=" * 60)

    print("   GPU Memory Types:")
    print("   1. Global Memory - Large, slow, accessible by all threads")
    print("   2. Shared Memory - Small, fast, shared within a block")
    print("   3. Registers     - Very fast, private to each thread")
    print("   4. Constant Memory - Cached global memory for constants")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\n   Current GPU Memory Info:")
        print(f"   Total Global Memory: {props.total_memory / 1e9:.1f} GB")
        print(
            f"   Shared Memory per Block: {props.shared_memory_per_block / 1024:.0f} KB"
        )
        print(f"   Registers per Multiprocessor: {props.regs_per_multiprocessor:,}")
        print(f"   Max Threads per MP: {props.max_threads_per_multi_processor}")


def demonstrate_performance_tips():
    """Demonstrate key performance optimization tips."""
    print("\n" + "=" * 60)
    print("PERFORMANCE OPTIMIZATION TIPS")
    print("=" * 60)

    print("   Key Performance Tips:")
    print("   1. Memory Coalescing - Access consecutive memory addresses")
    print("   2. Occupancy - Keep as many threads active as possible")
    print("   3. Minimize Divergence - Avoid if-statements when possible")
    print("   4. Use Shared Memory - Cache frequently accessed data")
    print("   5. Profile Your Code - Measure before optimizing")

    print("\n   Memory Access Patterns:")
    print("   Good: threads 0,1,2,3 access addresses 0,1,2,3 (coalesced)")
    print("   Bad:  threads 0,1,2,3 access addresses 0,2,4,6 (strided)")

    # Demonstrate the impact of memory access patterns
    if torch.cuda.is_available():
        N = 1_000_000
        x = torch.randn(N, device="cuda")

        print(f"\n   Memory Access Pattern Benchmark (N = {N:,}):")

        # Coalesced access (stride 1)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        y1 = x.clone()
        torch.cuda.synchronize()
        coalesced_time = (time.perf_counter() - start_time) * 1000

        # Strided access (stride 2)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        y2 = x[::2].clone()
        torch.cuda.synchronize()
        strided_time = (time.perf_counter() - start_time) * 1000

        print(f"   Coalesced (stride 1): {coalesced_time:.2f} ms")
        print(f"   Strided (stride 2):   {strided_time:.2f} ms")
        print(f"   Slowdown factor:      {strided_time/coalesced_time:.1f}x")


def main():
    """Main function to run all demonstrations."""
    print("  CUDA Programming Basics")
    print("This example teaches fundamental CUDA concepts")
    print("Through practical vector addition examples")

    demonstrate_cuda_basics()
    demonstrate_thread_organization()
    demonstrate_memory_concepts()
    demonstrate_performance_tips()

    print("\n" + "=" * 60)
    print("   CUDA Basics Complete!")
    print("Next steps:")
    print("   - Explore other examples (TBD)")
    print("   - Try modifying the vector size and block size")
    print("   - Run the benchmark suite: gpu-bench run memory")
    print("=" * 60)


if __name__ == "__main__":
    main()
