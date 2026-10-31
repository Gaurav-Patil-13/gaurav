#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

#define NUM_BINS 256
#define BLOCK_SIZE 256

// Version B: Global Memory Atomic Histogram
__global__ void histogram_global_atomic_kernel(const unsigned char *data, int *histogram, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        unsigned char val = data[id];
        atomicAdd(&histogram[val], 1);
    }
}

// Version C: Shared Memory Privatized Histogram
__global__ void histogram_shared_privatized_kernel(const unsigned char *data, int *histogram, int N) {
    __shared__ int local_hist[NUM_BINS];

    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Initialize shared memory sub-histogram to zero
    if (tid < NUM_BINS) {
        local_hist[tid] = 0;
    }
    __syncthreads();

    // Step 2: Accumulate local histogram in fast on-chip shared memory
    if (id < N) {
        unsigned char val = data[id];
        atomicAdd(&local_hist[val], 1);
    }
    __syncthreads();

    // Step 3: Commit private block histogram to global memory
    if (tid < NUM_BINS) {
        atomicAdd(&histogram[tid], local_hist[tid]);
    }
}

// Sequential CPU Histogram
void cpu_histogram(const unsigned char *data, int *histogram, int N) {
    for (int i = 0; i < NUM_BINS; i++) histogram[i] = 0;
    for (int i = 0; i < N; i++) {
        histogram[data[i]]++;
    }
}

int main() {
    printf("=================================================================\n");
    printf("     ASSIGNMENT 10 - TASK 6: PARALLEL CUDA HISTOGRAM COMPUTATION \n");
    printf("=================================================================\n");

    cudaFree(0);

    FILE *fp = fopen("histogram_benchmark.csv", "w");
    if (fp) {
        fprintf(fp, "Distribution,InputSize,CPUTimeMs,GlobalAtomicMs,SharedPrivatizedMs,SpeedupSharedOverGlobal,SpeedupSharedOverCPU\n");
    }

    int test_sizes[] = {1000000, 5000000, 10000000}; // 1M, 5M, 10M
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    const char *dist_names[] = {"Uniform Distribution", "Concentrated / Skewed Distribution"};

    for (int d = 0; d < 2; d++) {
        printf("\n=================================================================\n");
        printf(" EVALUATING: %s (Bins = %d)\n", dist_names[d], NUM_BINS);
        printf("=================================================================\n");
        printf("+--------------+-------------+------------------+------------------+-------------------------+-----------------+\n");
        printf("| Input Size   | CPU Time(ms)| Global Atomic(ms)| Shared Priv (ms) | Speedup (Shared/Global) | Speedup vs CPU  |\n");
        printf("+--------------+-------------+------------------+------------------+-------------------------+-----------------+\n");

        for (int s = 0; s < num_sizes; s++) {
            int N = test_sizes[s];
            unsigned char *h_data = (unsigned char*)malloc(N);

            // Populate according to distribution
            for (int i = 0; i < N; i++) {
                if (d == 0) {
                    h_data[i] = (unsigned char)(rand() % NUM_BINS); // Uniform across 0..255
                } else {
                    // Skewed: 85% of values hit only 4 bins (severe contention!)
                    if ((rand() % 100) < 85) {
                        h_data[i] = (unsigned char)(rand() % 4);
                    } else {
                        h_data[i] = (unsigned char)(rand() % NUM_BINS);
                    }
                }
            }

            int h_hist_cpu[NUM_BINS] = {0};
            int h_hist_gpu[NUM_BINS] = {0};

            // CPU Benchmark
            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_histogram(h_data, h_hist_cpu, N);
            auto t2 = std::chrono::high_resolution_clock::now();
            double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

            // GPU Buffers
            unsigned char *d_data;
            int *d_hist;
            cudaMalloc((void**)&d_data, N);
            cudaMalloc((void**)&d_hist, NUM_BINS * sizeof(int));
            cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);

            int threads = BLOCK_SIZE;
            int blocks = (N + threads - 1) / threads;

            cudaEvent_t start, stop;
            cudaEventCreate(&start); cudaEventCreate(&stop);

            // Run Global Atomic
            cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));
            cudaEventRecord(start);
            histogram_global_atomic_kernel<<<blocks, threads>>>(d_data, d_hist, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float glob_ms = 0.0f;
            cudaEventElapsedTime(&glob_ms, start, stop);

            // Run Shared Privatized
            cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));
            cudaEventRecord(start);
            histogram_shared_privatized_kernel<<<blocks, threads>>>(d_data, d_hist, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float priv_ms = 0.0f;
            cudaEventElapsedTime(&priv_ms, start, stop);

            cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

            // Verify
            bool match = true;
            for (int b = 0; b < NUM_BINS; b++) {
                if (h_hist_cpu[b] != h_hist_gpu[b]) {
                    match = false;
                    break;
                }
            }

            double speedup_shared = (double)glob_ms / priv_ms;
            double speedup_cpu = cpu_ms / priv_ms;

            printf("| %-12d | %11.2f | %16.3f | %16.3f | %21.2fx | %13.2fx |\n",
                   N, cpu_ms, glob_ms, priv_ms, speedup_shared, speedup_cpu);

            if (fp) {
                fprintf(fp, "%s,%d,%.3f,%.4f,%.4f,%.3f,%.3f\n",
                        (d == 0 ? "Uniform" : "Concentrated"), N, cpu_ms, glob_ms, priv_ms, speedup_shared, speedup_cpu);
            }

            printf("  -> Verification: %s\n", match ? "PASSED (100% exact match across all 256 bins)" : "FAILED");

            cudaFree(d_data); cudaFree(d_hist);
            cudaEventDestroy(start); cudaEventDestroy(stop);
            free(h_data);
        }
        printf("+--------------+-------------+------------------+------------------+-------------------------+-----------------+\n");
    }

    if (fp) fclose(fp);

    printf("\nAnalytical Answers:\n");
    printf("1. Why can atomic operations become a bottleneck?\n");
    printf("   -> Atomic operations serialize hardware memory accesses to the same address.\n");
    printf("      When thousands of threads update identical bins, parallel execution collapses into sequential order.\n");
    printf("2. What happens when many threads update the same bin?\n");
    printf("   -> Severe atomic collision and memory stall cycles degrade throughput significantly.\n");
    printf("3. How does shared memory reduce contention?\n");
    printf("   -> Privatization creates per-block histogram copies in low-latency SRAM.\n");
    printf("      Contention is isolated to threads within the block, and only one commit to global memory\n");
    printf("      occurs per block at the end.\n");
    printf("4. What happens when the number of histogram bins increases?\n");
    printf("   -> Collisions decrease because updates scatter across more addresses, but shared memory usage\n");
    printf("      per block increases, potentially limiting active block occupancy if bins exceed available SRAM.\n");

    return 0;
}
