# GPU Benchmark Suite Makefile

.PHONY: help install build test clean lint format docs examples

# default target
help:
	@echo "GPU Benchmark Suite - Available commands:"
	@echo ""
	@echo "  make install     - Install the package and dependencies"
	@echo "  make build       - Build CUDA kernels"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run code linting"
	@echo "  make format      - Format code with black and isort"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make docs        - Build documentation"
	@echo "  make examples    - Run example scripts"
	@echo "  make benchmark   - Run standard benchmarks"
	@echo ""

install:
	@echo "Installing GPU Benchmark Suite..."
	./install.sh

install-dev: install
	@echo "Installing development dependencies..."
	. .venv/bin/activate && pip install -e ".[dev,docs,profiling]"

# cuda
build:
	@echo "Building CUDA kernels..."
	. .venv/bin/activate && python setup_cuda.py build_ext --inplace

test:
	@echo "Running tests..."
	. .venv/bin/activate && pytest tests/ -v

test-cuda:
	@echo "Running CUDA tests..."
	. .venv/bin/activate && pytest tests/ -v -m cuda

test-coverage:
	@echo "Running tests with coverage..."
	. .venv/bin/activate && pytest tests/ --cov=gpu_benchmark --cov-report=html --cov-report=term

lint:
	@echo "Running linters..."
	. .venv/bin/activate && flake8 src/ tests/ examples/
	. .venv/bin/activate && mypy src/gpu_benchmark/

format:
	@echo "Formatting code..."
	. .venv/bin/activate && black src/ tests/ examples/
	. .venv/bin/activate && isort src/ tests/ examples/

check-format:
	@echo "Checking code format..."
	. .venv/bin/activate && black --check src/ tests/ examples/
	. .venv/bin/activate && isort --check-only src/ tests/ examples/

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -name "*.so" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

docs:
	@echo "Building documentation..."
	. .venv/bin/activate && cd docs && make html

examples:
	@echo "Running examples..."
	. .venv/bin/activate && cd examples/01_cuda_basics && python vector_add_example.py

benchmark:
	@echo "Running standard benchmarks..."
	. .venv/bin/activate && gpu-bench device-info
	. .venv/bin/activate && gpu-bench run memory --sizes 1024,4096,16384

benchmark-full:
	@echo "Running comprehensive benchmarks..."
	. .venv/bin/activate && gpu-bench run-all --sizes 1024,4096,16384,65536

profile:
	@echo "Profiling benchmarks..."
	. .venv/bin/activate && gpu-bench profile memory/vector-add cuda --size 4096
	. .venv/bin/activate && gpu-bench profile math/matmul-shared cuda --size 1024

setup-hooks:
	@echo "Setting up pre-commit hooks..."
	. .venv/bin/activate && pre-commit install

check-all:
	@echo "Running pre-commit on all files..."
	. .venv/bin/activate && pre-commit run --all-files

# dev workflow
dev: install-dev setup-hooks
	@echo "Development environment ready!"
	@echo "Next steps:"
	@echo "  source .venv/bin/activate"
	@echo "  make test"
	@echo "  make benchmark"

# CI workflow
ci: check-format lint test build
	@echo "CI checks passed!"

# Package for distribution
package: clean
	@echo "Building distribution packages..."
	. .venv/bin/activate && python -m build

# (test)
upload-test: package
	@echo "Uploading to TestPyPI..."
	. .venv/bin/activate && twine upload --repository testpypi dist/*

# (production)
upload: package
	@echo "Uploading to PyPI..."
	. .venv/bin/activate && twine upload dist/*

check-package: package
	@echo "Checking package..."
	. .venv/bin/activate && twine check dist/*

monitor:
	@echo "Monitoring GPU..."
	watch -n 1 nvidia-smi

reset: clean
	rm -rf .venv/
	make install

info:
	@echo "System Information"
	@echo "===================="
	@which python3
	@python3 --version
	@which nvcc && nvcc --version || echo "NVCC not found"
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "nvidia-smi not found"
