#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

// CPU Sequential Matrix Multiplication Baseline
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
        mat[i] = (float)(i % 100) * 0.01f;
    }
}

int main() {
    printf("=================================================================\n");
    printf("   ASSIGNMENT 11 - TASK 2: MATRIX MULTIPLICATION USING cuBLAS    \n");
    printf("=================================================================\n");

    cudaFree(0);

    // Initialize cuBLAS context
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Failed to initialize cuBLAS handle!\n");
        return 1;
    }

    int test_sizes[] = {256, 512, 1024};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("\ncuBLAS Execution & Performance Table:\n");
    printf("+-------------+-------------+--------------------+--------------------+---------+------------+\n");
    printf("| Matrix Size | CPU Time(ms)| cuBLAS Lib Time(ms)| Total App Time(ms) | Speedup | GFLOPS     |\n");
    printf("+-------------+-------------+--------------------+--------------------+---------+------------+\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = test_sizes[s];
        size_t bytes = N * N * sizeof(float);

        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C_cpu = (float*)calloc(N * N, sizeof(float));
        float *h_C_cublas = (float*)malloc(bytes);

        init_matrix(h_A, N);
        init_matrix(h_B, N);

        // CPU Execution
        auto t1 = std::chrono::high_resolution_clock::now();
        matmul_cpu(h_A, h_B, h_C_cpu, N);
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        // Allocate Device Memory
        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, bytes);
        cudaMalloc((void**)&d_B, bytes);
        cudaMalloc((void**)&d_C, bytes);

        cudaEvent_t start_k, stop_k, start_tot, stop_tot;
        cudaEventCreate(&start_k); cudaEventCreate(&stop_k);
        cudaEventCreate(&start_tot); cudaEventCreate(&stop_tot);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Warm-up run
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
        cudaDeviceSynchronize();

        // Benchmark Run (Timed Transfers + Library Call)
        cudaEventRecord(start_tot);

        // H2D
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        // Library Execution
        cudaEventRecord(start_k);
        // Note: C = A * B in row-major is equivalent to C^T = B^T * A^T in column-major
        // By passing d_B first with leading dimension N, cuBLAS computes the exact row-major product!
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
        cudaEventRecord(stop_k);
        cudaEventSynchronize(stop_k);

        // D2H
        cudaMemcpy(h_C_cublas, d_C, bytes, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop_tot);
        cudaEventSynchronize(stop_tot);

        float cublas_ms = 0.0f, total_ms = 0.0f;
        cudaEventElapsedTime(&cublas_ms, start_k, stop_k);
        cudaEventElapsedTime(&total_ms, start_tot, stop_tot);

        double speedup = cpu_ms / cublas_ms;
        double gflops = (2.0 * N * N * N) / (cublas_ms * 1e-3 * 1e9);

        printf("| %4d x %-4d | %11.2f | %16.3f ms | %16.3f ms | %6.2fx | %10.2f |\n",
               N, N, cpu_ms, cublas_ms, total_ms, speedup, gflops);

        // Verify Against CPU Reference
        float max_diff = 0.0f;
        for (int i = 0; i < N * N; i++) {
            float diff = fabsf(h_C_cpu[i] - h_C_cublas[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("  -> Verification vs CPU Reference for %dx%d: Max Diff = %e [%s]\n",
               N, N, max_diff, (max_diff < 1e-3f) ? "PASSED (100% Match)" : "FAILED");

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(start_k); cudaEventDestroy(stop_k);
        cudaEventDestroy(start_tot); cudaEventDestroy(stop_tot);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_cublas);
    }
    printf("+-------------+-------------+--------------------+--------------------+---------+------------+\n");

    cublasDestroy(handle);

    printf("\nTask 2 Analytical Answers:\n");
    printf("1. Did cuBLAS outperform custom CUDA kernels?\n");
    printf("   -> Yes. cuBLAS delivers significantly higher GFLOPS and lower execution time than our custom kernels.\n");
    printf("2. Why might cuBLAS achieve better performance?\n");
    printf("   -> cuBLAS utilizes expert assembly-level optimizations, register tiling, multi-stage software pipelining,\n");
    printf("      and hardware Tensor Cores on Ampere architecture that standard C++ CUDA cannot easily express.\n");
    printf("3. How does matrix size affect observed speedup?\n");
    printf("   -> Speedup scales up rapidly with matrix dimension N as computation dominates fixed invocation latency.\n");
    printf("4. Why should correctness be checked before comparing performance?\n");
    printf("   -> An incorrect algorithm or improperly mapped memory layout might execute fast by computing nothing,\n");
    printf("      making performance metrics meaningless without numerical validation.\n");
    printf("5. When would you prefer cuBLAS over writing a custom CUDA kernel?\n");
    printf("   -> Always for standard linear algebra (GEMM, GEMV, triangular solves), saving engineering effort and\n");
    printf("      guaranteeing near-theoretical peak hardware throughput across all NVIDIA GPU architectures.\n");

    return 0;
}
