# GPU Programming Learning Examples

This directory contains educational examples that demonstrate GPU programming concepts from basic to advanced. Each example is designed to teach specific concepts and includes detailed explanations.

## Learning Path

### 1. Basic Concepts
- `01_cuda_basics/` - Introduction to CUDA programming **[IMPLEMENTED]**
- `02_memory_hierarchy/` - Understanding GPU memory hierarchy **[TBD]**
- `03_thread_organization/` - Blocks, grids, and warps **[TBD]**

### 2. Memory Patterns
- `04_coalescing/` - Memory coalescing patterns **[TBD]**
- `05_shared_memory/` - Using shared memory effectively **[TBD]**
- `06_memory_bandwidth/` - Optimizing memory bandwidth **[TBD]**

### 3. Triton Fundamentals
- `07_triton_intro/` - Introduction to Triton **[TBD]**
- `08_triton_kernels/` - Writing Triton kernels **[TBD]**
- `09_triton_vs_cuda/` - Comparing Triton and CUDA **[TBD]**

### 4. Optimization Techniques
- `10_register_usage/` - Managing register usage **[TBD]**
- `11_occupancy/` - Understanding and optimizing occupancy **[TBD]**
- `12_profiling/` - Profiling and debugging techniques **[TBD]**

### 5. Advanced Patterns
- `13_cooperative_groups/` - Using cooperative groups **[TBD]**
- `14_warp_primitives/` - Warp-level primitives **[TBD]**
- `15_tensor_cores/` - Programming Tensor Cores **[TBD]**

## Current Implementation Status

### Implemented Examples
- **01_cuda_basics**: Complete CUDA introduction with vector addition example, detailed comments, and educational content

### To Be Developed (TBD)
- **14 additional examples** covering memory hierarchy, optimization techniques, Triton programming, and advanced GPU patterns
- **Interactive Jupyter notebooks** for hands-on learning
- **Performance comparison studies** between different implementations
- **Profiling and debugging tutorials** with real-world examples


### Currently Available (01_cuda_basics)
1. Begin with `01_cuda_basics` for CUDA fundamentals
2. The example includes runnable code with detailed comments
3. Modify parameters and observe performance changes
4. Use the benchmark suite to compare CUDA, Triton, and PyTorch implementations


## Prerequisites

- NVIDIA GPU with Compute Capability 7.0+
- CUDA Toolkit 11.8+
- Python 3.11+
- Basic understanding of parallel programming concepts

## Getting Started

```bash
# Navigate to the available example directory
cd examples/01_cuda_basics

# Run the CUDA basics example
python vector_add_example.py

# Note: Jupyter notebook mentioned below is TBD
# jupyter notebook cuda_basics.ipynb
```

**Note**: Only the `01_cuda_basics` example is currently implemented. The jupyter notebook and additional examples are planned for future development.
