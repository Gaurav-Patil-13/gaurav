#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

// CUDA Kernel for Vector Addition
__global__ void vector_add_kernel(const float *A, const float *B, float *C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary check to prevent out-of-bounds memory accesses
    if (id < N) {
        C[id] = A[id] + B[id];
    }
}

// Sequential CPU Vector Addition
void vector_add_cpu(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[]) {
    int N = 1000000; // 1 Million elements default
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("=================================================================\n");
    printf("       ASSIGNMENT 9 - TASK 4: VECTOR ADDITION USING CUDA         \n");
    printf("=================================================================\n");
    printf("Vector size N: %d elements (%.2f MB per array)\n\n", N, (N * sizeof(float)) / (1024.0 * 1024.0));

    size_t size_bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size_bytes);
    float *h_B = (float*)malloc(size_bytes);
    float *h_C_cpu = (float*)malloc(size_bytes);
    float *h_C_gpu = (float*)malloc(size_bytes);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(i % 100) * 0.5f;
        h_B[i] = (float)((i + 1) % 100) * 1.5f;
    }

    // Step 1: Sequential CPU Execution
    printf("[Step 1] Executing Sequential Vector Addition on CPU...\n");
    auto t1 = std::chrono::high_resolution_clock::now();
    vector_add_cpu(h_A, h_B, h_C_cpu, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("  Sequential CPU Execution Time: %.4f ms\n\n", cpu_time);

    // Step 2: CUDA GPU Execution
    printf("[Step 2] Executing Vector Addition on GPU using CUDA Kernel...\n");

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_bytes);
    cudaMalloc((void**)&d_B, size_bytes);
    cudaMalloc((void**)&d_C, size_bytes);

    cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    printf("  Kernel launch configuration: %d blocks, %d threads/block (Total threads: %d)\n",
           blocks_per_grid, threads_per_block, blocks_per_grid * threads_per_block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_kernel_time = 0.0f;
    cudaEventElapsedTime(&gpu_kernel_time, start, stop);

    cudaMemcpy(h_C_gpu, d_C, size_bytes, cudaMemcpyDeviceToHost);
    printf("  CUDA GPU Kernel Execution Time: %.4f ms\n\n", gpu_kernel_time);

    // Correctness Verification
    printf("[Step 3] Correctness Verification (CPU Result vs GPU Result):\n");
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("  Sample index 0:   CPU = %.2f, GPU = %.2f\n", h_C_cpu[0], h_C_gpu[0]);
    printf("  Sample index 100: CPU = %.2f, GPU = %.2f\n", h_C_cpu[100], h_C_gpu[100]);
    printf("  Sample index %d: CPU = %.2f, GPU = %.2f\n", N - 1, h_C_cpu[N - 1], h_C_gpu[N - 1]);
    printf("  Maximum absolute difference: %e\n", max_diff);
    if (max_diff < 1e-5f) {
        printf("  Verification Result: PASSED (CPU and GPU outputs match perfectly within tolerance)\n");
    } else {
        printf("  Verification Result: FAILED\n");
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
