"""
Main benchmark runner orchestrating different benchmark implementations.
"""

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .device_info import DeviceInfo
from .metrics import BenchmarkResult, MetricsCollector


class BenchmarkRunner:
    """Main benchmark runner class."""

    def __init__(self, device_id: int = 0):
        """Initialize benchmark runner.

        Args:
            device_id: GPU device ID to use for benchmarks
        """
        self.device_id = device_id
        self.device_info = DeviceInfo()
        self.metrics_collector = MetricsCollector(device_id)
        self._registered_benchmarks: Dict[str, Dict[str, Any]] = {}

        if not self.device_info.cuda_available:
            raise RuntimeError("CUDA is not available. GPU benchmarks require CUDA.")

        # load available benchmarks
        self._load_benchmarks()

    def _load_benchmarks(self) -> None:
        """Load and register available benchmark implementations."""
        benchmark_categories = ["memory", "math", "matrix", "advanced"]

        for category in benchmark_categories:
            self._registered_benchmarks[category] = {}

        # load benchmark modules and register them
        try:
            from ..benchmarks.memory import register_memory_benchmarks

            register_memory_benchmarks(self)
        except ImportError as e:
            print(f"Warning: Could not load memory benchmarks: {e}")

        try:
            from ..benchmarks.math import register_math_benchmarks

            register_math_benchmarks(self)
        except ImportError as e:
            print(f"Warning: Could not load math benchmarks: {e}")

    def register_benchmark(
        self,
        category: str,
        name: str,
        cuda_impl: Optional[callable] = None,
        triton_impl: Optional[callable] = None,
        pytorch_impl: Optional[callable] = None,
        flops_counter: Optional[callable] = None,
        bandwidth_counter: Optional[callable] = None,
    ) -> None:
        """Register a benchmark with different backend implementations.

        Args:
            category: Benchmark category (memory, math, etc.)
            name: Benchmark name
            cuda_impl: CUDA implementation function
            triton_impl: Triton implementation function
            pytorch_impl: PyTorch implementation function
            flops_counter: Function to compute FLOPS for the operation
            bandwidth_counter: Function to compute memory bandwidth
        """
        if category not in self._registered_benchmarks:
            self._registered_benchmarks[category] = {}

        self._registered_benchmarks[category][name] = {
            "cuda": cuda_impl,
            "triton": triton_impl,
            "pytorch": pytorch_impl,
            "flops_counter": flops_counter,
            "bandwidth_counter": bandwidth_counter,
        }

    def list_benchmarks(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """List available benchmarks.

        Args:
            category: Specific category to list, or None for all

        Returns:
            Dictionary mapping categories to benchmark names
        """
        if category:
            if category in self._registered_benchmarks:
                return {category: list(self._registered_benchmarks[category].keys())}
            else:
                return {}

        return {
            cat: list(benchmarks.keys())
            for cat, benchmarks in self._registered_benchmarks.items()
            if benchmarks
        }

    def run_benchmark(
        self,
        category: str,
        name: str,
        backends: List[str] = None,
        sizes: List[int] = None,
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
        **kwargs,
    ) -> List[BenchmarkResult]:
        """Run a specific benchmark across backends.

        Args:
            category: Benchmark category
            name: Benchmark name
            backends: List of backends to test (cuda, triton, pytorch)
            sizes: List of problem sizes to test
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
            **kwargs: Additional arguments passed to benchmark functions

        Returns:
            List of BenchmarkResult objects
        """
        if category not in self._registered_benchmarks:
            raise ValueError(f"Unknown benchmark category: {category}")

        if name not in self._registered_benchmarks[category]:
            raise ValueError(f"Unknown benchmark: {name} in category {category}")

        if backends is None:
            backends = ["cuda", "triton", "pytorch"]

        if sizes is None:
            sizes = [1024, 4096, 16384]  # Default sizes

        benchmark_def = self._registered_benchmarks[category][name]
        results = []

        for size in sizes:
            for backend in backends:
                impl_func = benchmark_def.get(backend)
                if impl_func is None:
                    print(f"Warning: No {backend} implementation for {name}")
                    continue

                try:
                    # prepare arguments for the benchmark
                    args, ref_output = self._prepare_benchmark_args(
                        category, name, backend, size, **kwargs
                    )

                    # run benchmark
                    result = self.metrics_collector.benchmark_function(
                        func=impl_func,
                        name=f"{name}_size_{size}",
                        backend=backend,
                        args=args,
                        warmup_runs=warmup_runs,
                        benchmark_runs=benchmark_runs,
                        compute_flops=benchmark_def.get("flops_counter"),
                        compute_bandwidth=benchmark_def.get("bandwidth_counter"),
                        reference_output=ref_output,
                    )

                    # add size information to metadata
                    result.metadata["size"] = size
                    result.metadata["category"] = category
                    results.append(result)

                except Exception as e:
                    print(f"Error running {backend} {name} (size {size}): {e}")

        return results

    def _prepare_benchmark_args(
        self, category: str, name: str, backend: str, size: int, **kwargs
    ) -> Tuple[tuple, Optional[Any]]:
        """Prepare arguments for benchmark functions.

        This method creates appropriate test data based on the benchmark type.

        Args:
            category: Benchmark category
            name: Benchmark name
            backend: Backend being tested
            size: Problem size
            **kwargs: Additional parameters

        Returns:
            Tuple of (args, reference_output)
        """
        import torch

        # default argument preparation based on common patterns
        if "vector" in name or "element" in name:
            # vector operations - two input vectors
            a = torch.randn(size, device=f"cuda:{self.device_id}", dtype=torch.float32)
            b = torch.randn(size, device=f"cuda:{self.device_id}", dtype=torch.float32)
            return (a, b), None

        elif "matrix" in name or "matmul" in name:
            # Matrix operations
            sqrt_size = int(size**0.5)
            a = torch.randn(
                sqrt_size,
                sqrt_size,
                device=f"cuda:{self.device_id}",
                dtype=torch.float32,
            )
            b = torch.randn(
                sqrt_size,
                sqrt_size,
                device=f"cuda:{self.device_id}",
                dtype=torch.float32,
            )
            return (a, b), None

        elif "reduce" in name or "sum" in name:
            # Reduction operations
            a = torch.randn(size, device=f"cuda:{self.device_id}", dtype=torch.float32)
            return (a,), None

        else:
            # Default: single vector
            a = torch.randn(size, device=f"cuda:{self.device_id}", dtype=torch.float32)
            return (a,), None

    def run_category(
        self,
        category: str,
        backends: List[str] = None,
        sizes: List[int] = None,
        **kwargs,
    ) -> List[BenchmarkResult]:
        """Run all benchmarks in a category.

        Args:
            category: Benchmark category to run
            backends: List of backends to test
            sizes: List of problem sizes to test
            **kwargs: Additional arguments

        Returns:
            List of all BenchmarkResult objects
        """
        if category not in self._registered_benchmarks:
            raise ValueError(f"Unknown benchmark category: {category}")

        results = []
        for benchmark_name in self._registered_benchmarks[category]:
            try:
                bench_results = self.run_benchmark(
                    category, benchmark_name, backends, sizes, **kwargs
                )
                results.extend(bench_results)
            except Exception as e:
                print(f"Error running benchmark {benchmark_name}: {e}")

        return results

    def run_all(
        self, backends: List[str] = None, sizes: List[int] = None, **kwargs
    ) -> List[BenchmarkResult]:
        """Run all available benchmarks.

        Args:
            backends: List of backends to test
            sizes: List of problem sizes to test
            **kwargs: Additional arguments

        Returns:
            List of all BenchmarkResult objects
        """
        results = []
        for category in self._registered_benchmarks:
            if self._registered_benchmarks[category]:  # only if category has benchmarks
                try:
                    cat_results = self.run_category(category, backends, sizes, **kwargs)
                    results.extend(cat_results)
                except Exception as e:
                    print(f"Error running category {category}: {e}")

        return results

    def profile_benchmark(
        self, category: str, name: str, backend: str, size: int = 4096, **kwargs
    ) -> None:
        """Profile a specific benchmark using NVIDIA profiling tools.

        Args:
            category: Benchmark category
            name: Benchmark name
            backend: Backend to profile
            size: Problem size
            **kwargs: Additional arguments
        """
        # To-do: integrate with NVIDIA Nsight or other profiling tools
        print(
            f"Profiling {backend} implementation of {category}/{name} with size {size}"
        )
        print("Note: Profiling integration to be implemented")

        # For now, just run the benchmark with detailed timing
        results = self.run_benchmark(
            category, name, [backend], [size], warmup_runs=1, benchmark_runs=1, **kwargs
        )

        if results:
            result = results[0]
            print(f"Execution time: {result.execution_time:.3f} ms")
            if result.memory_used:
                print(f"Memory used: {result.memory_used:.1f} MB")
            if result.flops:
                print(f"Performance: {result.flops:.1f} GFLOPS")

    def get_device_info(self) -> None:
        """Print device information."""
        self.device_info.print_device_info(self.device_id)

    def clear_results(self) -> None:
        """Clear all collected results."""
        self.metrics_collector.clear_results()

    def get_results(self) -> List[BenchmarkResult]:
        """Get all collected results."""
        return self.metrics_collector.get_results()

    def print_results(self) -> None:
        """Print formatted results."""
        self.metrics_collector.print_results()

    def compare_results(
        self, baseline_backend: str = "pytorch"
    ) -> Dict[str, Dict[str, float]]:
        """Compare results across backends."""
        return self.metrics_collector.compare_results(baseline_backend)
