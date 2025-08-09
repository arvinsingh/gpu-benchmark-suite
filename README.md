# GPU Benchmark Suite

A comprehensive CLI tool for benchmarking GPU performance across CUDA, Triton, and PyTorch implementations. Designed as both a practical benchmarking tool and an educational resource for learning GPU programming and optimization.

## Features

- **Multi-Backend Benchmarking** - Compare performance across CUDA C++, Triton, and PyTorch
- **Educational Examples** - Learn GPU programming concepts with progressively complex examples
- **Comprehensive Metrics** - Memory bandwidth, FLOPS, latency, and throughput analysis
- **Configurable Benchmarks** - Test various problem sizes, data types, and configurations
- **Extensible Architecture** - Easy to add new kernels and benchmarks

## Benchmark Categories

### 1. Memory Operations
- Vector addition, multiplication
- Memory bandwidth tests
- Coalesced vs non-coalesced memory access patterns

### 2. Mathematical Operations
- Matrix multiplication (naive and shared memory algorithms)
- Reduction operations (sum)
- Element-wise operations (ReLU, GELU, Softmax)

## Installation

```bash
# Install from source
git clone <repo-url>
cd gpu-benchmark-suite
pip install -e .

# Or using uv
uv pip install -e .
```

## Quick Start

```bash
# Run all benchmarks
gpu-bench run-all

# Run specific category
gpu-bench run memory

# Compare implementations
gpu-bench compare vector-add --sizes 1024,4096,16384

# Profile a specific kernel
gpu-bench profile matmul-cuda --size 1024
```

## Project Structure

```
gpu-benchmark-suite/
├── src/gpu_benchmark/
│   ├── cli/                  # CLI interface
│   ├── benchmarks/           # Benchmark implementations
│   │   ├── memory.py         # Memory benchmarks
│   │   ├── math.py           # Math benchmarks
│   │   ├── cuda/             # Cuda wrapper implementation*
│   │   ├── triton/           # Triton implementation*
│   │   └── pytorch/          # PyTorch implementation*
│   │   # *New benchmarks will use this structure for cleaner organization
│   ├── core/                 # Core benchmark infrastructure
│   ├── metrics/              # Performance metrics collection (stub)
│   ├── profiling/            # Profiling and analysis tools (stub)
│   └── utils/                # Utilities and helpers (empty)
├── kernels/                  # CUDA source files
├── examples/                 # Educational examples
├── tests/                    # Basic test suite
└── docs/                     # Documentation
```

## Current Implementation Status

### Working Features
- Memory Benchmarks - vector addition, multiplication, memory copy, strided access
- Math Benchmarks - matrix multiplication (naive & shared memory), reduction sum, activation functions (ReLU, GELU, Softmax)
- Multi-Backend Support - CUDA, Triton, and PyTorch implementations for core benchmarks
- CLI Interface - full command-line interface with device info, benchmark listing, running, and comparison
- Educational Examples - basics tutorial in examples directory

### Partial Implementation
- Performance Profiling - Basic profiling command available but limited functionality
- Metrics Collection - Core infrastructure present but detailed analysis TBD

### Planned Features
- Advanced kernels (convolutions, attention mechanisms)
- NVIDIA Nsight integration for detailed profiling
- Memory hierarchy analysis tools
- Cache optimization benchmarks

## To Be Developed (TBD)

### Advanced Kernels
- Convolutions
- Attention mechanisms
- Custom compute patterns

### Memory Patterns
- Shared memory utilization
- Cache optimization
- Memory hierarchy analysis

### Mathematical Operations Extensions
- Reduction operations (max, min, mean, etc.)
- Advanced matrix operations (decompositions, solvers)
- FFT and signal processing kernels

### Performance Analysis
- NVIDIA Nsight integration for detailed profiling
- Memory access pattern analysis
- Occupancy optimization tools
- Energy consumption benchmarking

### Educational Content
- Additional tutorial examples beyond basics
- Interactive Jupyter notebooks for each concept
- Performance optimization workshops

## Learning Path

1. Explore `examples/` directory for comprehensive learning materials. See [examples/README.md](examples/README.md) for the complete learning path from basic concepts to advanced optimization techniques.
2. Run Basic Benchmarks; begin with memory operations (`memory/vector-add`)
3. Use `compare` command to understand trade-offs between CUDA, Triton, and PyTorch.
4. Try matrix multiplication and reduction benchmarks and explore math ops.
5. Add your own kernels following existing patterns

Note: Currently only `01_cuda_basics` example is implemented. Additional examples listed in examples/README.md are planned for future development. :)

## Requirements

- NVIDIA GPU (Compute Capability 7.0+)
- CUDA Toolkit 11.8+
- Python 3.11+
- PyTorch 2.0+
- Triton 2.0+

