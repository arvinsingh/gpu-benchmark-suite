/*
Python bindings for CUDA kernels using PyTorch C++ extensions.
*/

#include <torch/extension.h>
#include <vector>

// Include CUDA kernel headers
extern "C" {
    void launch_vector_add(const float* a, const float* b, float* c, int n);
    void launch_vector_mul(const float* a, const float* b, float* c, int n);
    void launch_memory_copy(const float* input, float* output, int n);
    void launch_strided_access(const float* input, float* output, int n, int stride);
    void launch_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K);
    void launch_matmul_shared(const float* A, const float* B, float* C, int M, int N, int K);
    void launch_reduce_sum(const float* input, float* output, int n);
    void launch_reduce_sum_shared(const float* input, float* output, int n);
}

// helper function to check tensor properties
void check_tensor(const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.device().type() == torch::kCUDA, name + " must be on CUDA device");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, name + " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name + " must be contiguous");
}

// =============================================================================
// Vector Operations
// =============================================================================

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    check_tensor(a, "a");
    check_tensor(b, "b");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");

    auto c = torch::empty_like(a);
    int n = a.numel();

    launch_vector_add(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    return c;
}

torch::Tensor vector_mul_cuda(torch::Tensor a, torch::Tensor b) {
    check_tensor(a, "a");
    check_tensor(b, "b");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");

    auto c = torch::empty_like(a);
    int n = a.numel();

    launch_vector_mul(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    return c;
}

// =============================================================================
// Memory Operations
// =============================================================================

torch::Tensor memory_copy_cuda(torch::Tensor input) {
    check_tensor(input, "input");

    auto output = torch::empty_like(input);
    int n = input.numel();

    launch_memory_copy(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

torch::Tensor strided_access_cuda(torch::Tensor input, int stride) {
    check_tensor(input, "input");

    int n = input.numel() / stride;
    auto output = torch::empty({n}, torch::dtype(torch::kFloat32).device(input.device()));

    launch_strided_access(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        stride
    );

    return output;
}

// =============================================================================
// Matrix Operations
// =============================================================================

torch::Tensor matmul_naive_cuda(torch::Tensor A, torch::Tensor B) {
    check_tensor(A, "A");
    check_tensor(B, "B");

    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must be compatible");

    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

    launch_matmul_naive(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

torch::Tensor matmul_shared_cuda(torch::Tensor A, torch::Tensor B) {
    check_tensor(A, "A");
    check_tensor(B, "B");

    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must be compatible");

    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

    launch_matmul_shared(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

// =============================================================================
// Reduction Operations
// =============================================================================

torch::Tensor reduce_sum_cuda(torch::Tensor input) {
    check_tensor(input, "input");

    int n = input.numel();
    int blocks = (n + 255) / 256;  // BLOCK_SIZE = 256

    auto output = torch::empty({blocks}, torch::dtype(torch::kFloat32).device(input.device()));

    launch_reduce_sum(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // final reduction on CPU if multiple blocks
    if (blocks > 1) {
        return output.sum();
    } else {
        return output[0];
    }
}

torch::Tensor reduce_sum_shared_cuda(torch::Tensor input) {
    check_tensor(input, "input");

    int n = input.numel();
    int blocks = std::min(64, (n + 255) / 256);  // BLOCK_SIZE = 256

    auto output = torch::empty({blocks}, torch::dtype(torch::kFloat32).device(input.device()));

    launch_reduce_sum_shared(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    if (blocks > 1) {
        return output.sum();
    } else {
        return output[0];
    }
}

// =============================================================================
// Python Module Definition
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GPU Benchmark Suite CUDA kernels";

    // Vector operations
    m.def("vector_add", &vector_add_cuda, "CUDA vector addition");
    m.def("vector_mul", &vector_mul_cuda, "CUDA vector multiplication");

    // Memory operations
    m.def("memory_copy", &memory_copy_cuda, "CUDA memory copy");
    m.def("strided_access", &strided_access_cuda, "CUDA strided memory access");

    // Matrix operations
    m.def("matmul_naive", &matmul_naive_cuda, "CUDA naive matrix multiplication");
    m.def("matmul_shared", &matmul_shared_cuda, "CUDA shared memory matrix multiplication");

    // Reduction operations
    m.def("reduce_sum", &reduce_sum_cuda, "CUDA sum reduction");
    m.def("reduce_sum_shared", &reduce_sum_shared_cuda, "CUDA sum reduction with shared memory");
}
