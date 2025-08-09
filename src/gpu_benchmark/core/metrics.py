"""
Performance metrics collection and analysis utilities.
"""

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    backend: str
    execution_time: float  # milliseconds
    memory_used: Optional[float] = None  # MB
    throughput: Optional[float] = None  # operations per second
    bandwidth: Optional[float] = None  # GB/s
    flops: Optional[float] = None  # GFLOPS
    accuracy: Optional[float] = None  # for comparison with reference
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MetricsCollector:
    """Collect and analyze performance metrics."""

    def __init__(self, device_id: int = 0):
        """Initialize metrics collector.

        Args:
            device_id: GPU device ID to monitor
        """
        self.device_id = device_id
        self.device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )
        self._results: List[BenchmarkResult] = []

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB.

        Returns:
            Memory usage in MB
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / 1024**2
        return 0.0

    def get_memory_info(self) -> Dict[str, float]:
        """Get detailed memory information.

        Returns:
            Dictionary with memory statistics in MB
        """
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated": torch.cuda.memory_allocated(self.device) / 1024**2,
            "cached": torch.cuda.memory_reserved(self.device) / 1024**2,
            "max_allocated": torch.cuda.max_memory_allocated(self.device) / 1024**2,
            "max_cached": torch.cuda.max_memory_reserved(self.device) / 1024**2,
        }

    @contextmanager
    def time_execution(self):
        """Context manager for timing GPU operations."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            start_time = time.perf_counter()

            yield

            end_event.record()
            torch.cuda.synchronize(self.device)
            gpu_time = start_event.elapsed_time(end_event)  # milliseconds
            cpu_time = (time.perf_counter() - start_time) * 1000  # milliseconds

            # Store both GPU and CPU timing
            self._last_gpu_time = gpu_time
            self._last_cpu_time = cpu_time
        else:
            start_time = time.perf_counter()
            yield
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # milliseconds
            self._last_gpu_time = execution_time
            self._last_cpu_time = execution_time

    def benchmark_function(
        self,
        func: Callable,
        name: str,
        backend: str,
        args: tuple = (),
        kwargs: dict = None,
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
        compute_flops: Optional[Callable] = None,
        compute_bandwidth: Optional[Callable] = None,
        reference_output: Optional[torch.Tensor] = None,
    ) -> BenchmarkResult:
        """Benchmark a function with comprehensive metrics.

        Args:
            func: Function to benchmark
            name: Benchmark name
            backend: Backend identifier (cuda, triton, pytorch)
            args: Function arguments
            kwargs: Function keyword arguments
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
            compute_flops: Function to compute FLOPS given input sizes
            compute_bandwidth: Function to compute bandwidth given input sizes
            reference_output: Reference output for accuracy comparison

        Returns:
            BenchmarkResult with collected metrics
        """
        if kwargs is None:
            kwargs = {}

        # Clear cache and collect garbage
        self.clear_cache()

        # Warmup runs
        for _ in range(warmup_runs):
            with self.time_execution():
                output = func(*args, **kwargs)

        # Benchmark runs
        times = []
        memory_usage_start = self.get_memory_usage()

        for _ in range(benchmark_runs):
            self.clear_cache()
            memory_before = self.get_memory_usage()

            with self.time_execution():
                output = func(*args, **kwargs)

            times.append(self._last_gpu_time)

        memory_usage_end = self.get_memory_usage()
        memory_used = max(0, memory_usage_end - memory_usage_start)

        # Calculate statistics
        execution_time = np.mean(times)

        # Calculate FLOPS if function provided
        flops = None
        if compute_flops:
            try:
                ops_per_second = compute_flops(*args, **kwargs)
                flops = ops_per_second / (execution_time / 1000) / 1e9  # GFLOPS
            except Exception:
                flops = None

        # Calculate bandwidth if function provided
        bandwidth = None
        if compute_bandwidth:
            try:
                bytes_per_run = compute_bandwidth(*args, **kwargs)
                bandwidth = bytes_per_run / (execution_time / 1000) / 1e9  # GB/s
            except Exception:
                bandwidth = None

        # Calculate throughput (operations per second)
        throughput = 1000 / execution_time  # ops/second

        # Calculate accuracy if reference provided
        accuracy = None
        if reference_output is not None and output is not None:
            try:
                if isinstance(output, torch.Tensor) and isinstance(
                    reference_output, torch.Tensor
                ):
                    diff = torch.abs(output - reference_output)
                    max_diff = torch.max(diff).item()
                    mean_diff = torch.mean(diff).item()
                    accuracy = {"max_error": max_diff, "mean_error": mean_diff}
            except Exception:
                accuracy = None

        # create result
        result = BenchmarkResult(
            name=name,
            backend=backend,
            execution_time=execution_time,
            memory_used=memory_used if memory_used > 0 else None,
            throughput=throughput,
            bandwidth=bandwidth,
            flops=flops,
            accuracy=accuracy,
            metadata={
                "warmup_runs": warmup_runs,
                "benchmark_runs": benchmark_runs,
                "execution_times": times,
                "std_dev": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "device": str(self.device),
            },
        )

        self._results.append(result)
        return result

    def get_results(self) -> List[BenchmarkResult]:
        """Get all collected benchmark results."""
        return self._results.copy()

    def clear_results(self) -> None:
        """Clear all collected results."""
        self._results.clear()

    def compare_results(
        self, baseline_backend: str = "pytorch"
    ) -> Dict[str, Dict[str, float]]:
        """Compare results across different backends.

        Args:
            baseline_backend: Backend to use as baseline for speedup calculation

        Returns:
            Dictionary with comparison metrics
        """
        # group results by benchmark name
        grouped = {}
        for result in self._results:
            if result.name not in grouped:
                grouped[result.name] = {}
            grouped[result.name][result.backend] = result

        comparisons = {}
        for bench_name, backends in grouped.items():
            if baseline_backend not in backends:
                continue

            baseline = backends[baseline_backend]
            comparisons[bench_name] = {}

            for backend_name, result in backends.items():
                if backend_name == baseline_backend:
                    speedup = 1.0
                else:
                    speedup = baseline.execution_time / result.execution_time

                comparisons[bench_name][backend_name] = {
                    "execution_time": result.execution_time,
                    "speedup": speedup,
                    "memory_used": result.memory_used,
                    "throughput": result.throughput,
                    "flops": result.flops,
                    "bandwidth": result.bandwidth,
                }

        return comparisons

    def print_results(self) -> None:
        """Print formatted benchmark results."""
        if not self._results:
            print("No benchmark results available.")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        # group by benchmark name
        grouped = {}
        for result in self._results:
            if result.name not in grouped:
                grouped[result.name] = []
            grouped[result.name].append(result)

        for bench_name, results in grouped.items():
            print(f"\n{bench_name.upper()}")
            print("-" * 40)

            # sort by execution time
            results.sort(key=lambda x: x.execution_time)

            for result in results:
                print(f"{result.backend:>10}: {result.execution_time:>8.3f} ms", end="")

                if result.memory_used:
                    print(f" | {result.memory_used:>6.1f} MB", end="")

                if result.flops:
                    print(f" | {result.flops:>6.1f} GFLOPS", end="")

                if result.bandwidth:
                    print(f" | {result.bandwidth:>6.1f} GB/s", end="")

                print()

            # show speedups relative to slowest
            if len(results) > 1:
                slowest_time = results[-1].execution_time
                print("\nSpeedup vs slowest:")
                for result in results:
                    speedup = slowest_time / result.execution_time
                    print(f"{result.backend:>10}: {speedup:>6.2f}x")

        print("=" * 80)
