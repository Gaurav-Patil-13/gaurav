#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void vector_add_kernel_opt(const float *A, const float *B, float *C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        C[id] = A[id] + B[id];
    }
}

void vector_add_seq(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

void vector_add_omp(const float *A, const float *B, float *C, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    printf("=================================================================\n");
    printf("  ASSIGNMENT 9 - TASK 7: SEQUENTIAL vs OpenMP vs CUDA BENCHMARK \n");
    printf("=================================================================\n");

    // CUDA Driver Context Warm-up (standard HPC practice)
    cudaFree(0);
    float *d_dummy;
    cudaMalloc((void**)&d_dummy, sizeof(float));
    vector_add_kernel_opt<<<1, 1>>>(d_dummy, d_dummy, d_dummy, 1);
    cudaDeviceSynchronize();
    cudaFree(d_dummy);

    int test_sizes[] = {100000, 1000000, 10000000}; // 10^5, 10^6, 10^7
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    // Open CSV file for Graph 1 and Graph 3 plotting
    FILE *fp = fopen("task7_comparative_benchmark.csv", "w");
    if (fp) {
        fprintf(fp, "VectorSize,SequentialMs,OpenMPMs,CUDAKernelMs,CUDAH2DMs,CUDAD2HMs,CUDATotalMs,SpeedupKernel,SpeedupTotal\n");
    }

    printf("Benchmarking across problem sizes:\n\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = test_sizes[s];
        size_t bytes = N * sizeof(float);
        printf("-----------------------------------------------------------------\n");
        printf(">>> Problem Size N = %d (10^%d elements, %.2f MB per array)\n",
               N, (int)log10(N), bytes / (1024.0 * 1024.0));
        printf("-----------------------------------------------------------------\n");

        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C_seq = (float*)malloc(bytes);
        float *h_C_omp = (float*)malloc(bytes);
        float *h_C_cuda = (float*)malloc(bytes);

        for (int i = 0; i < N; i++) {
            h_A[i] = (float)(i % 100) * 0.25f;
            h_B[i] = (float)((i + 2) % 100) * 0.75f;
        }

        // 1. Sequential Execution
        auto t_seq_start = std::chrono::high_resolution_clock::now();
        vector_add_seq(h_A, h_B, h_C_seq, N);
        auto t_seq_end = std::chrono::high_resolution_clock::now();
        double seq_ms = std::chrono::duration<double, std::milli>(t_seq_end - t_seq_start).count();

        // 2. OpenMP Execution
        auto t_omp_start = std::chrono::high_resolution_clock::now();
        vector_add_omp(h_A, h_B, h_C_omp, N);
        auto t_omp_end = std::chrono::high_resolution_clock::now();
        double omp_ms = std::chrono::duration<double, std::milli>(t_omp_end - t_omp_start).count();

        // 3. CUDA Execution with Component Breakdown
        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, bytes);
        cudaMalloc((void**)&d_B, bytes);
        cudaMalloc((void**)&d_C, bytes);

        cudaEvent_t h2d_start, h2d_stop, kern_start, kern_stop, d2h_start, d2h_stop;
        cudaEventCreate(&h2d_start); cudaEventCreate(&h2d_stop);
        cudaEventCreate(&kern_start); cudaEventCreate(&kern_stop);
        cudaEventCreate(&d2h_start); cudaEventCreate(&d2h_stop);

        // Host to Device Transfer
        cudaEventRecord(h2d_start);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(h2d_stop);
        cudaEventSynchronize(h2d_stop);
        float h2d_ms = 0.0f;
        cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop);

        // Kernel Execution
        int threads_per_block = 256;
        int blocks = (N + threads_per_block - 1) / threads_per_block;
        cudaEventRecord(kern_start);
        vector_add_kernel_opt<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(kern_stop);
        cudaEventSynchronize(kern_stop);
        float kern_ms = 0.0f;
        cudaEventElapsedTime(&kern_ms, kern_start, kern_stop);

        // Device to Host Transfer
        cudaEventRecord(d2h_start);
        cudaMemcpy(h_C_cuda, d_C, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(d2h_stop);
        cudaEventSynchronize(d2h_stop);
        float d2h_ms = 0.0f;
        cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop);

        float cuda_total_ms = h2d_ms + kern_ms + d2h_ms;

        // Correctness Verification
        float max_diff_omp = 0.0f, max_diff_cuda = 0.0f;
        for (int i = 0; i < N; i++) {
            float d1 = fabsf(h_C_seq[i] - h_C_omp[i]);
            float d2 = fabsf(h_C_seq[i] - h_C_cuda[i]);
            if (d1 > max_diff_omp) max_diff_omp = d1;
            if (d2 > max_diff_cuda) max_diff_cuda = d2;
        }

        double speedup_omp = seq_ms / omp_ms;
        double speedup_kern = seq_ms / kern_ms;
        double speedup_total = seq_ms / cuda_total_ms;

        printf("  Sequential CPU Time:        %10.4f ms\n", seq_ms);
        printf("  OpenMP Multi-Core Time:     %10.4f ms  (Speedup: %.2fx)\n", omp_ms, speedup_omp);
        printf("  CUDA Host-to-Device (H2D):  %10.4f ms\n", h2d_ms);
        printf("  CUDA Kernel Execution:      %10.4f ms  (Kernel Speedup: %.2fx)\n", kern_ms, speedup_kern);
        printf("  CUDA Device-to-Host (D2H):  %10.4f ms\n", d2h_ms);
        printf("  CUDA Total End-to-End:      %10.4f ms  (End-to-End Speedup: %.2fx)\n", cuda_total_ms, speedup_total);
        printf("  Verification Status:        OMP diff=%e, CUDA diff=%e [%s]\n\n",
               max_diff_omp, max_diff_cuda, (max_diff_cuda < 1e-5f) ? "PASSED" : "FAILED");

        if (fp) {
            fprintf(fp, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    N, seq_ms, omp_ms, kern_ms, h2d_ms, d2h_ms, cuda_total_ms, speedup_kern, speedup_total);
        }

        // Clean up
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(h2d_start); cudaEventDestroy(h2d_stop);
        cudaEventDestroy(kern_start); cudaEventDestroy(kern_stop);
        cudaEventDestroy(d2h_start); cudaEventDestroy(d2h_stop);
        free(h_A); free(h_B); free(h_C_seq); free(h_C_omp); free(h_C_cuda);
    }

    if (fp) fclose(fp);

    printf("=================================================================\n");
    printf(" ANALYSIS & DISCUSSION ANSWERS                                   \n");
    printf("=================================================================\n");
    printf("1. Which implementation is fastest for smaller input sizes (10^5)?\n");
    printf("   -> OpenMP or Sequential CPU is often fastest for small inputs because\n");
    printf("      the PCIe data transfer overhead and CUDA runtime launch latency dominate\n");
    printf("      the small computational workload.\n\n");
    printf("2. What happens as the input size increases (10^6, 10^7)?\n");
    printf("   -> CUDA GPU kernel speedup increases dramatically. The sheer number of\n");
    printf("      CUDA cores (over 2000 hardware execution threads) provides massive memory\n");
    printf("      bandwidth and computation throughput that easily surpasses CPU cores.\n\n");
    printf("3. Why might CUDA have overhead for small problems?\n");
    printf("   -> CUDA kernel launch latency (~5-10 microseconds), thread block scheduling,\n");
    printf("      and PCIe memory copying (H2D and D2H) introduce fixed latency costs that\n");
    printf("      exceed the tiny compute time of small arrays.\n\n");
    printf("4. What additional overhead is introduced by host-device data transfer?\n");
    printf("   -> Host-to-Device and Device-to-Host transfers must traverse the PCIe bus\n");
    printf("      (typically 8-16 GB/s), which is orders of magnitude slower than on-chip GPU\n");
    printf("      VRAM bandwidth (over 100-200 GB/s).\n\n");
    printf("5. Why can CUDA become advantageous for highly parallel workloads?\n");
    printf("   -> For large data parallel workloads, thousands of GPU threads operate in parallel,\n");
    printf("      overlapping memory latency via warp scheduling and delivering tens of times\n");
    printf("      higher throughput than multi-core CPUs.\n");

    return 0;
}
