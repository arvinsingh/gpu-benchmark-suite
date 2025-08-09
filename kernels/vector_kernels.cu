#include "vector_kernels.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define TILE_SIZE 16

// utility function for error checking
extern "C" void check_cuda_error(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// =============================================================================
// Vector Operations
// =============================================================================

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_mul_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// =============================================================================
// Memory Operations
// =============================================================================

__global__ void memory_copy_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

__global__ void strided_access_kernel(const float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx * stride];
    }
}

// =============================================================================
// Matrix Operations
// =============================================================================

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// Reduction Operations
// =============================================================================

__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

__global__ void reduce_sum_shared_kernel(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// =============================================================================
// Host wrapper functions (launcher functions for Python bindings)
// =============================================================================

extern "C" {
    
void launch_vector_add(const float* a, const float* b, float* c, int n) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    vector_add_kernel<<<grid, block>>>(a, b, c, n);
    cudaDeviceSynchronize();
    check_cuda_error("vector_add_kernel");
}

void launch_vector_mul(const float* a, const float* b, float* c, int n) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    vector_mul_kernel<<<grid, block>>>(a, b, c, n);
    cudaDeviceSynchronize();
    check_cuda_error("vector_mul_kernel");
}

void launch_memory_copy(const float* input, float* output, int n) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    memory_copy_kernel<<<grid, block>>>(input, output, n);
    cudaDeviceSynchronize();
    check_cuda_error("memory_copy_kernel");
}

void launch_strided_access(const float* input, float* output, int n, int stride) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    strided_access_kernel<<<grid, block>>>(input, output, n, stride);
    cudaDeviceSynchronize();
    check_cuda_error("strided_access_kernel");
}

void launch_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
    check_cuda_error("matmul_naive_kernel");
}

void launch_matmul_shared(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_shared_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
    check_cuda_error("matmul_shared_kernel");
}

void launch_reduce_sum(const float* input, float* output, int n) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int shared_mem_size = BLOCK_SIZE * sizeof(float);
    
    reduce_sum_kernel<<<grid, block, shared_mem_size>>>(input, output, n);
    cudaDeviceSynchronize();
    check_cuda_error("reduce_sum_kernel");
}

void launch_reduce_sum_shared(const float* input, float* output, int n) {
    int blocks = min(64, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    dim3 grid(blocks);
    
    reduce_sum_shared_kernel<<<grid, block>>>(input, output, n);
    cudaDeviceSynchronize();
    check_cuda_error("reduce_sum_shared_kernel");
}

} // extern "C"
