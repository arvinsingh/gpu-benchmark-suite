#!/bin/bash

# GPU Benchmark Suite Installation Script
# This script sets up the complete development environment

set -e

echo "GPU Benchmark Suite Installation"
echo "=================================="

# colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # no Color

# print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_cuda() {
    print_status "Checking CUDA installation..."
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        print_status "CUDA version: $CUDA_VERSION"
        
        if command -v nvidia-smi &> /dev/null; then
            DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
            print_status "NVIDIA Driver version: $DRIVER_VERSION"
        else
            print_warning "nvidia-smi not found"
        fi
    else
        print_error "CUDA not found! Please install CUDA toolkit 11.8 or later"
        echo "Visit: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
}

check_python() {
    print_status "Checking Python version..."
    
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        print_status "Python version: $PYTHON_VERSION ✓"
    else
        print_error "Python 3.11+ required, found: $PYTHON_VERSION"
        exit 1
    fi
}

install_system_deps() {
    print_status "Installing system dependencies..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y build-essential python3-dev python3-pip
        elif command -v yum &> /dev/null; then
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y python3-devel python3-pip
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install python@3.11
        else
            print_warning "Homebrew not found. Please install manually."
        fi
    fi
}

setup_python_env() {
    print_status "Setting up Python environment..."
    
    # check if uv available
    if command -v uv &> /dev/null; then
        print_status "Using uv for package management"
        uv venv --python=python3.11
        source .venv/bin/activate
        uv pip install -e .
    else
        print_status "Using pip for package management"
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip setuptools wheel
        pip install -e .
    fi
}

build_cuda_kernels() {
    print_status "Building CUDA kernels..."
    
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # build CUDA extensions
    python setup_cuda.py build_ext --inplace
    
    if [ $? -eq 0 ]; then
        print_status "CUDA kernels built successfully ✓"
    else
        print_warning "CUDA kernel build failed - continuing with PyTorch/Triton only"
    fi
}

install_dev_deps() {
    print_status "Installing development dependencies..."
    
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    if command -v uv &> /dev/null; then
        uv pip install -e ".[dev,docs,profiling]"
    else
        pip install -e ".[dev,docs,profiling]"
    fi
}

run_tests() {
    print_status "Running tests..."
    
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # run basic tests (excluding CUDA-specific ones if no GPU)
    pytest tests/ -v --tb=short
    
    if [ $? -eq 0 ]; then
        print_status "Tests passed ✓"
    else
        print_warning "Some tests failed - check the output above"
    fi
}

setup_pre_commit() {
    print_status "Setting up pre-commit hooks..."
    
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    pre-commit install
    print_status "Pre-commit hooks installed ✓"
}

verify_installation() {
    print_status "Verifying installation..."
    
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # CLI test
    if gpu-bench --help &> /dev/null; then
        print_status "CLI working ✓"
    else
        print_error "CLI installation failed"
        return 1
    fi
    
    # test Python import
    python -c "import gpu_benchmark; print('✓ Package import successful')"

    # test device detection
    python -c "
from gpu_benchmark.core.device_info import DeviceInfo
info = DeviceInfo()
if info.cuda_available:
    print('✓ CUDA available')
    info.print_device_info(0)
else:
    print('⚠ CUDA not available - limited functionality')
"
}

# show usage information
show_usage() {
    echo ""
    echo "Installation Complete!"
    echo "========================"
    echo ""
    echo "To get started:"
    echo "  source .venv/bin/activate  # Activate virtual environment"
    echo "  gpu-bench device-info      # Check GPU information"
    echo "  gpu-bench list-benchmarks  # List available benchmarks"
    echo "  gpu-bench run memory       # Run memory benchmarks"
    echo ""
    echo "Examples:"
    echo "  cd examples/01_cuda_basics && python vector_add_example.py"
    echo ""
    echo "Documentation:"
    echo "  See README.md and docs/ for detailed information"
    echo "  Use 'gpu-bench --help' for CLI usage"
    echo ""
}

# installation flow
main() {
    echo "Starting installation process..."
    echo ""
    
    check_python
    check_cuda
    install_system_deps
    setup_python_env
    build_cuda_kernels
    install_dev_deps
    setup_pre_commit
    run_tests
    verify_installation
    show_usage
}

# parse command line args
case "${1:-install}" in
    "install"|"")
        main
        ;;
    "cuda")
        check_cuda
        build_cuda_kernels
        ;;
    "test")
        run_tests
        ;;
    "verify")
        verify_installation
        ;;
    "help"|"--help"|"-h")
        echo "Usage: $0 [install|cuda|test|verify|help]"
        echo ""
        echo "Commands:"
        echo "  install  - Full installation (default)"
        echo "  cuda     - Build CUDA kernels only"
        echo "  test     - Run tests only"
        echo "  verify   - Verify installation only"
        echo "  help     - Show this help"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
