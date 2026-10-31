#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

// CUDA Kernel: Matrix Multiplication using Global Memory
// Each thread computes one element C[row, col]
__global__ void matmul_global_kernel(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Sequential CPU Matrix Multiplication Baseline
void matmul_cpu(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

void init_matrix(float *mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = (float)(i % 100) / 100.0f;
    }
}

int main() {
    printf("=================================================================\n");
    printf(" ASSIGNMENT 10 - TASK 1: MATRIX MULTIPLICATION (GLOBAL MEMORY)   \n");
    printf("=================================================================\n");

    // Warm up CUDA driver
    cudaFree(0);

    int test_sizes[] = {256, 512, 1024};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    int block_configs[][2] = {{8, 8}, {16, 16}, {32, 32}};
    int num_configs = sizeof(block_configs) / sizeof(block_configs[0]);

    for (int s = 0; s < num_sizes; s++) {
        int N = test_sizes[s];
        size_t bytes = N * N * sizeof(float);
        printf("\n-----------------------------------------------------------------\n");
        printf(" Matrix Dimension: %d x %d (%.2f MB per matrix)\n", N, N, bytes / (1024.0 * 1024.0));
        printf("-----------------------------------------------------------------\n");

        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C_cpu = (float*)calloc(N * N, sizeof(float));
        float *h_C_gpu = (float*)malloc(bytes);

        init_matrix(h_A, N);
        init_matrix(h_B, N);

        // CPU Execution
        printf(" Running CPU sequential matrix multiplication...\n");
        auto t1 = std::chrono::high_resolution_clock::now();
        matmul_cpu(h_A, h_B, h_C_cpu, N);
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double gflops_cpu = (2.0 * N * N * N) / (cpu_time * 1e-3 * 1e9);
        printf("  CPU Execution Time: %8.2f ms | Throughput: %6.2f GFLOPS\n\n", cpu_time, gflops_cpu);

        // Allocate Device Buffers
        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, bytes);
        cudaMalloc((void**)&d_B, bytes);
        cudaMalloc((void**)&d_C, bytes);

        // Test across 2D Block Configurations
        printf(" Running CUDA Global Memory Kernel across block configurations:\n");
        printf(" +---------------+---------------+------------------+------------------+---------+---------+\n");
        printf(" | Block Dim     | Grid Dim      | Kernel Time (ms) | Total Time (ms)  | Speedup | GFLOPS  |\n");
        printf(" +---------------+---------------+------------------+------------------+---------+---------+\n");

        cudaEvent_t start_k, stop_k, start_tot, stop_tot;
        cudaEventCreate(&start_k); cudaEventCreate(&stop_k);
        cudaEventCreate(&start_tot); cudaEventCreate(&stop_tot);

        for (int c = 0; c < num_configs; c++) {
            dim3 block(block_configs[c][0], block_configs[c][1]);
            dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

            cudaEventRecord(start_tot);

            // H2D
            cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

            // Timed Kernel
            cudaEventRecord(start_k);
            matmul_global_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
            cudaEventRecord(stop_k);
            cudaEventSynchronize(stop_k);

            // D2H
            cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

            cudaEventRecord(stop_tot);
            cudaEventSynchronize(stop_tot);

            float kern_ms = 0.0f, total_ms = 0.0f;
            cudaEventElapsedTime(&kern_ms, start_k, stop_k);
            cudaEventElapsedTime(&total_ms, start_tot, stop_tot);

            double speedup_kern = cpu_time / kern_ms;
            double gflops_gpu = (2.0 * N * N * N) / (kern_ms * 1e-3 * 1e9);

            printf(" | %2dx%-11d | %3dx%-10d | %16.3f | %16.3f | %6.2fx | %7.2f |\n",
                   block.x, block.y, grid.x, grid.y, kern_ms, total_ms, speedup_kern, gflops_gpu);
        }
        printf(" +---------------+---------------+------------------+------------------+---------+---------+\n");

        // Verify Correctness
        float max_diff = 0.0f;
        for (int i = 0; i < N * N; i++) {
            float diff = fabsf(h_C_cpu[i] - h_C_gpu[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf(" Numerical Verification: Max Diff = %e [%s]\n",
               max_diff, (max_diff < 1e-3f) ? "PASSED" : "FAILED");

        // Cleanup
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(start_k); cudaEventDestroy(stop_k);
        cudaEventDestroy(start_tot); cudaEventDestroy(stop_tot);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    }

    return 0;
}
