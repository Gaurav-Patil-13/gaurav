#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

// Option B: Element-Wise Vector Multiplication Kernel: C[i] = A[i] * B[i]
__global__ void elementwise_mult_kernel(const float *A, const float *B, float *C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        C[id] = A[id] * B[id];
    }
}

// Option C: Array Squaring Kernel: B[i] = A[i] * A[i]
__global__ void array_square_kernel(const float *A, float *B, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        B[id] = A[id] * A[id];
    }
}

// Sequential CPU Implementations
void elementwise_mult_cpu(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] * B[i];
    }
}

void array_square_cpu(const float *A, float *B, int N) {
    for (int i = 0; i < N; i++) {
        B[i] = A[i] * A[i];
    }
}

int main(int argc, char *argv[]) {
    int N = 5000000; // 5 Million elements
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    size_t bytes = N * sizeof(float);

    printf("=================================================================\n");
    printf(" ASSIGNMENT 9 - TASK 8: 1D ARRAY PROCESSING (ELEMENT-WISE MULT & SQUARE)\n");
    printf("=================================================================\n");
    printf("Problem Size N = %d elements (%.2f MB per array)\n\n", N, bytes / (1024.0 * 1024.0));

    // CUDA Driver Context Warm-up
    cudaFree(0);
    float *d_dummy;
    cudaMalloc((void**)&d_dummy, sizeof(float));
    elementwise_mult_kernel<<<1, 1>>>(d_dummy, d_dummy, d_dummy, 1);
    cudaDeviceSynchronize();
    cudaFree(d_dummy);

    // Allocate Host Memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);
    float *h_Sq_cpu = (float*)malloc(bytes);
    float *h_Sq_gpu = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)((i % 50) + 1) * 0.1f;
        h_B[i] = (float)(((i + 3) % 50) + 1) * 0.2f;
    }

    // -----------------------------------------------------------------
    // Part 1: Option B - Element-wise Vector Multiplication
    // -----------------------------------------------------------------
    printf(">>> PART 1: OPTION B - Element-wise Vector Multiplication (C = A * B)\n");

    // CPU Execution
    auto t1 = std::chrono::high_resolution_clock::now();
    elementwise_mult_cpu(h_A, h_B, h_C_cpu, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_time_mult = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // GPU Execution
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaEvent_t h2d_start, h2d_stop, kern_start, kern_stop, d2h_start, d2h_stop;
    cudaEventCreate(&h2d_start); cudaEventCreate(&h2d_stop);
    cudaEventCreate(&kern_start); cudaEventCreate(&kern_stop);
    cudaEventCreate(&d2h_start); cudaEventCreate(&d2h_stop);

    // H2D
    cudaEventRecord(h2d_start);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_stop);
    cudaEventSynchronize(h2d_stop);
    float h2d_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop);

    // Kernel
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    cudaEventRecord(kern_start);
    elementwise_mult_kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(kern_stop);
    cudaEventSynchronize(kern_stop);
    float kern_ms = 0.0f;
    cudaEventElapsedTime(&kern_ms, kern_start, kern_stop);

    // D2H
    cudaEventRecord(d2h_start);
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(d2h_stop);
    cudaEventSynchronize(d2h_stop);
    float d2h_ms = 0.0f;
    cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop);

    float gpu_total_ms = h2d_ms + kern_ms + d2h_ms;

    // Verify
    float max_diff_mult = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = fabsf(h_C_cpu[i] - h_C_gpu[i]);
        if (d > max_diff_mult) max_diff_mult = d;
    }

    printf("  Sequential CPU Time:            %10.4f ms\n", cpu_time_mult);
    printf("  CUDA Host-to-Device Transfer:   %10.4f ms\n", h2d_ms);
    printf("  CUDA Kernel Execution Time:     %10.4f ms\n", kern_ms);
    printf("  CUDA Device-to-Host Transfer:   %10.4f ms\n", d2h_ms);
    printf("  CUDA Total Execution Time:      %10.4f ms\n", gpu_total_ms);
    printf("  Kernel Speedup (T_cpu/T_kern):  %10.2fx\n", cpu_time_mult / kern_ms);
    printf("  End-to-End Speedup (T_cpu/T_gpu):%9.2fx\n", cpu_time_mult / gpu_total_ms);
    printf("  Max Absolute Difference:        %e [%s]\n\n",
           max_diff_mult, (max_diff_mult < 1e-5f) ? "VERIFICATION PASSED" : "FAILED");

    // -----------------------------------------------------------------
    // Part 2: Option C - Array Squaring (B = A^2)
    // -----------------------------------------------------------------
    printf(">>> PART 2: OPTION C - Array Squaring (B[i] = A[i]^2)\n");

    auto t3 = std::chrono::high_resolution_clock::now();
    array_square_cpu(h_A, h_Sq_cpu, N);
    auto t4 = std::chrono::high_resolution_clock::now();
    double cpu_time_sq = std::chrono::duration<double, std::milli>(t4 - t3).count();

    cudaEventRecord(kern_start);
    array_square_kernel<<<blocks, threads_per_block>>>(d_A, d_C, N);
    cudaEventRecord(kern_stop);
    cudaEventSynchronize(kern_stop);
    float kern_sq_ms = 0.0f;
    cudaEventElapsedTime(&kern_sq_ms, kern_start, kern_stop);

    cudaMemcpy(h_Sq_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    float max_diff_sq = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = fabsf(h_Sq_cpu[i] - h_Sq_gpu[i]);
        if (d > max_diff_sq) max_diff_sq = d;
    }

    printf("  Sequential CPU Time:            %10.4f ms\n", cpu_time_sq);
    printf("  CUDA Kernel Execution Time:     %10.4f ms\n", kern_sq_ms);
    printf("  Kernel Speedup:                 %10.2fx\n", cpu_time_sq / kern_sq_ms);
    printf("  Max Absolute Difference:        %e [%s]\n\n",
           max_diff_sq, (max_diff_sq < 1e-5f) ? "VERIFICATION PASSED" : "FAILED");

    // Clean up
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(h2d_start); cudaEventDestroy(h2d_stop);
    cudaEventDestroy(kern_start); cudaEventDestroy(kern_stop);
    cudaEventDestroy(d2h_start); cudaEventDestroy(d2h_stop);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu); free(h_Sq_cpu); free(h_Sq_gpu);

    return 0;
}
