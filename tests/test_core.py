"""
Tests for the GPU benchmark suite core functionality.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from gpu_benchmark.core.device_info import DeviceInfo
from gpu_benchmark.core.metrics import MetricsCollector, BenchmarkResult
from gpu_benchmark.core.benchmark_runner import BenchmarkRunner


class TestDeviceInfo:
    """Tests for DeviceInfo class."""
    
    @pytest.fixture
    def device_info(self):
        return DeviceInfo()
    
    def test_cuda_availability(self, device_info):
        """Test CUDA availability detection."""
        # This should match torch.cuda.is_available()
        assert isinstance(device_info.cuda_available, bool)
        assert device_info.cuda_available == torch.cuda.is_available()
    
    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_count(self, device_info):
        """Test device count detection."""
        assert device_info.device_count > 0
        assert device_info.device_count == torch.cuda.device_count()
    
    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_info(self, device_info):
        """Test device information retrieval."""
        info = device_info.get_device_info(0)
        
        # Check required fields
        assert "device_id" in info
        assert "name" in info
        assert "compute_capability" in info
        assert "total_memory" in info
        assert "multi_processor_count" in info
        
        # Check data types
        assert isinstance(info["device_id"], int)
        assert isinstance(info["name"], str)
        assert isinstance(info["total_memory"], int)
        assert info["total_memory"] > 0
    
    def test_get_system_info(self, device_info):
        """Test system information retrieval."""
        info = device_info.get_system_info()
        
        assert "platform" in info
        assert "python_version" in info
        assert "pytorch_version" in info
        
        assert isinstance(info["platform"], str)
        assert isinstance(info["pytorch_version"], str)
    
    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_invalid_device_id(self, device_info):
        """Test error handling for invalid device ID."""
        invalid_id = device_info.device_count + 10
        
        with pytest.raises(ValueError, match="Invalid device ID"):
            device_info.get_device_info(invalid_id)


class TestMetricsCollector:
    """Tests for MetricsCollector class."""
    
    @pytest.fixture
    def metrics_collector(self):
        return MetricsCollector(device_id=0)
    
    def test_initialization(self, metrics_collector):
        """Test MetricsCollector initialization."""
        assert metrics_collector.device_id == 0
        assert len(metrics_collector.get_results()) == 0
    
    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_memory_usage(self, metrics_collector):
        """Test memory usage measurement."""
        # Initial memory usage should be non-negative
        initial_usage = metrics_collector.get_memory_usage()
        assert initial_usage >= 0
        
        # Allocate tensor and check memory increases
        x = torch.randn(1000, 1000, device='cuda')
        new_usage = metrics_collector.get_memory_usage()
        assert new_usage > initial_usage
        
        del x
        torch.cuda.empty_cache()
    
    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_time_execution_context_manager(self, metrics_collector):
        """Test timing context manager."""
        def dummy_kernel():
            x = torch.randn(1000, device='cuda')
            y = torch.randn(1000, device='cuda')
            return x + y
        
        with metrics_collector.time_execution():
            result = dummy_kernel()
        
        # Check that timing was recorded
        assert hasattr(metrics_collector, '_last_gpu_time')
        assert metrics_collector._last_gpu_time > 0
        
        del result
    
    def test_benchmark_function(self, metrics_collector):
        """Test function benchmarking."""
        def simple_add(a, b):
            return a + b
        
        # Create test data
        if torch.cuda.is_available():
            a = torch.randn(1000, device='cuda')
            b = torch.randn(1000, device='cuda')
        else:
            a = torch.randn(1000)
            b = torch.randn(1000)
        
        # Run benchmark
        result = metrics_collector.benchmark_function(
            func=simple_add,
            name="test_add",
            backend="test",
            args=(a, b),
            warmup_runs=1,
            benchmark_runs=3
        )
        
        # Check result
        assert isinstance(result, BenchmarkResult)
        assert result.name == "test_add"
        assert result.backend == "test"
        assert result.execution_time > 0
        assert len(result.metadata["execution_times"]) == 3
    
    def test_clear_results(self, metrics_collector):
        """Test clearing benchmark results."""
        # Add a dummy result
        result = BenchmarkResult("test", "backend", 1.0)
        metrics_collector._results.append(result)
        
        assert len(metrics_collector.get_results()) == 1
        
        metrics_collector.clear_results()
        assert len(metrics_collector.get_results()) == 0


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""
    
    def test_creation(self):
        """Test BenchmarkResult creation."""
        result = BenchmarkResult(
            name="test_benchmark",
            backend="pytorch",
            execution_time=1.5,
            memory_used=100.0,
            throughput=1000.0
        )
        
        assert result.name == "test_benchmark"
        assert result.backend == "pytorch"
        assert result.execution_time == 1.5
        assert result.memory_used == 100.0
        assert result.throughput == 1000.0
        assert result.metadata == {}  # Default value
    
    def test_default_metadata(self):
        """Test default metadata initialization."""
        result = BenchmarkResult("test", "backend", 1.0)
        assert result.metadata == {}
        
        result_with_metadata = BenchmarkResult(
            "test", "backend", 1.0, 
            metadata={"key": "value"}
        )
        assert result_with_metadata.metadata == {"key": "value"}


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""
    
    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_initialization(self):
        """Test BenchmarkRunner initialization."""
        runner = BenchmarkRunner(device_id=0)
        
        assert runner.device_id == 0
        assert isinstance(runner.device_info, DeviceInfo)
        assert isinstance(runner.metrics_collector, MetricsCollector)
        assert isinstance(runner._registered_benchmarks, dict)
    
    def test_initialization_no_cuda(self):
        """Test BenchmarkRunner initialization without CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                BenchmarkRunner()
    
    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_register_benchmark(self):
        """Test benchmark registration."""
        runner = BenchmarkRunner()
        
        def dummy_impl(x):
            return x * 2
        
        runner.register_benchmark(
            category="test",
            name="dummy",
            pytorch_impl=dummy_impl
        )
        
        assert "test" in runner._registered_benchmarks
        assert "dummy" in runner._registered_benchmarks["test"]
        assert runner._registered_benchmarks["test"]["dummy"]["pytorch"] == dummy_impl
    
    @pytest.mark.cuda  
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_list_benchmarks(self):
        """Test benchmark listing."""
        runner = BenchmarkRunner()
        
        # Register a test benchmark
        runner.register_benchmark("test", "dummy", pytorch_impl=lambda x: x)
        
        benchmarks = runner.list_benchmarks()
        assert isinstance(benchmarks, dict)
        
        # Should include our test benchmark
        if "test" in benchmarks:
            assert "dummy" in benchmarks["test"]
        
        # Test category-specific listing
        test_benchmarks = runner.list_benchmarks("test")
        assert "test" in test_benchmarks
        assert "dummy" in test_benchmarks["test"]
    
    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prepare_benchmark_args(self):
        """Test benchmark argument preparation."""
        runner = BenchmarkRunner()
        
        # Test vector operation
        args, ref = runner._prepare_benchmark_args("memory", "vector-add", "pytorch", 1000)
        assert len(args) == 2  # Two input vectors
        assert isinstance(args[0], torch.Tensor)
        assert isinstance(args[1], torch.Tensor)
        assert args[0].shape == (1000,)
        
        # Test matrix operation
        args, ref = runner._prepare_benchmark_args("math", "matmul", "pytorch", 1024)
        assert len(args) == 2  # Two matrices
        assert isinstance(args[0], torch.Tensor)
        assert isinstance(args[1], torch.Tensor)
        assert args[0].dim() == 2  # 2D tensor
        
        # Test reduction operation
        args, ref = runner._prepare_benchmark_args("math", "reduce-sum", "pytorch", 1000)
        assert len(args) == 1  # Single input
        assert isinstance(args[0], torch.Tensor)
        assert args[0].shape == (1000,)


@pytest.fixture
def sample_benchmark_results():
    """Fixture providing sample benchmark results."""
    return [
        BenchmarkResult("test1", "pytorch", 10.0, flops=100.0),
        BenchmarkResult("test1", "cuda", 5.0, flops=200.0),
        BenchmarkResult("test2", "pytorch", 20.0, flops=50.0),
        BenchmarkResult("test2", "cuda", 8.0, flops=125.0),
    ]


def test_comparison(sample_benchmark_results):
    """Test result comparison functionality."""
    collector = MetricsCollector()
    collector._results = sample_benchmark_results
    
    comparison = collector.compare_results("pytorch")
    
    # Check structure
    assert "test1" in comparison
    assert "test2" in comparison
    assert "pytorch" in comparison["test1"]
    assert "cuda" in comparison["test1"]
    
    # Check speedups
    assert comparison["test1"]["cuda"]["speedup"] == 2.0  # 10.0 / 5.0
    assert comparison["test2"]["cuda"]["speedup"] == 2.5  # 20.0 / 8.0
    
    # Check baseline speedup is 1.0
    assert comparison["test1"]["pytorch"]["speedup"] == 1.0
    assert comparison["test2"]["pytorch"]["speedup"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
