#ifndef VECTOR_KERNELS_CUH
#define VECTOR_KERNELS_CUH

// C++ compatible header - only declares launcher functions

extern "C" {
    // Vector operations - matching the cpp file declarations
    void launch_vector_add(const float* a, const float* b, float* c, int n);
    void launch_vector_mul(const float* a, const float* b, float* c, int n);
    
    // Memory operations
    void launch_memory_copy(const float* input, float* output, int n);
    void launch_strided_access(const float* input, float* output, int n, int stride);
    
    // Matrix operations
    void launch_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K);
    void launch_matmul_shared(const float* A, const float* B, float* C, int M, int N, int K);
    
    // Reduction operations
    void launch_reduce_sum(const float* input, float* output, int n);
    void launch_reduce_sum_shared(const float* input, float* output, int n);
    
    // Utility functions
    void check_cuda_error(const char* message);
}

// Constants for use in C++
#define BLOCK_SIZE 256
#define TILE_SIZE 16

#endif // VECTOR_KERNELS_CUH
