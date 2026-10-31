#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <chrono>

#define BLOCK_SIZE 256

// Version B: Global Memory Atomic Reduction
__global__ void reduce_global_atomic_sum(const int *d_in, int *d_out, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        atomicAdd(d_out, d_in[id]);
    }
}

// Version C: Shared Memory Tree-Based Reduction (Sum)
__global__ void reduce_shared_tree_sum(const int *d_in, int *d_out, int N) {
    __shared__ int sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sdata[tid] = (id < N) ? d_in[id] : 0;
    __syncthreads();

    // Logarithmic in-place tree reduction in shared memory (strided to avoid warp divergence)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 atomically commits the block's partial sum to global memory
    if (tid == 0) {
        atomicAdd(d_out, sdata[0]);
    }
}

// Shared Memory Tree-Based Reduction for Maximum
__global__ void reduce_shared_tree_max(const int *d_in, int *d_out, int N) {
    __shared__ int sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (id < N) ? d_in[id] : INT_MIN;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(d_out, sdata[0]);
    }
}

// Sequential CPU Reduction
void cpu_reduction(const int *data, int N, int *sum, int *max_val, int *min_val) {
    int s = 0;
    int mx = INT_MIN;
    int mn = INT_MAX;
    for (int i = 0; i < N; i++) {
        s += data[i];
        if (data[i] > mx) mx = data[i];
        if (data[i] < mn) mn = data[i];
    }
    *sum = s;
    *max_val = mx;
    *min_val = mn;
}

int main() {
    printf("=================================================================\n");
    printf("     ASSIGNMENT 10 - TASK 4: PARALLEL REDUCTION (SUM/MAX/MIN)    \n");
    printf("=================================================================\n");

    cudaFree(0);

    FILE *fp = fopen("reduction_benchmark.csv", "w");
    if (fp) {
        fprintf(fp, "ArraySize,CPUTimeMs,GlobalAtomicMs,SharedTreeMs,SpeedupSharedOverCPU,SpeedupSharedOverGlobal\n");
    }

    int test_sizes[] = {100000, 1000000, 10000000}; // 10^5, 10^6, 10^7
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("\nExecution Comparison Table:\n");
    printf("+--------------+-------------+------------------+------------------+------------------+-----------------+\n");
    printf("| Array Size   | CPU Time(ms)| Global Atomic(ms)| Shared Tree (ms) | Speedup vs CPU   | Speedup vs Glob |\n");
    printf("+--------------+-------------+------------------+------------------+------------------+-----------------+\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = test_sizes[s];
        size_t bytes = N * sizeof(int);

        int *h_in = (int*)malloc(bytes);
        for (int i = 0; i < N; i++) {
            h_in[i] = (i % 10) + 1; // values 1..10
        }

        // CPU Reduction
        int cpu_sum = 0, cpu_max = 0, cpu_min = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_reduction(h_in, N, &cpu_sum, &cpu_max, &cpu_min);
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        // GPU Buffers
        int *d_in, *d_out;
        cudaMalloc((void**)&d_in, bytes);
        cudaMalloc((void**)&d_out, sizeof(int));
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

        int threads = BLOCK_SIZE;
        int blocks = (N + threads - 1) / threads;

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // Version B: Global Atomic Reduction
        int zero = 0;
        cudaMemcpy(d_out, &zero, sizeof(int), cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        reduce_global_atomic_sum<<<blocks, threads>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float glob_ms = 0.0f;
        cudaEventElapsedTime(&glob_ms, start, stop);

        int glob_res = 0;
        cudaMemcpy(&glob_res, d_out, sizeof(int), cudaMemcpyDeviceToHost);

        // Version C: Shared Memory Tree Reduction
        cudaMemcpy(d_out, &zero, sizeof(int), cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        reduce_shared_tree_sum<<<blocks, threads>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float shared_ms = 0.0f;
        cudaEventElapsedTime(&shared_ms, start, stop);

        int shared_res = 0;
        cudaMemcpy(&shared_res, d_out, sizeof(int), cudaMemcpyDeviceToHost);

        // Test Extended Reduction (Maximum)
        int init_max = INT_MIN;
        cudaMemcpy(d_out, &init_max, sizeof(int), cudaMemcpyHostToDevice);
        reduce_shared_tree_max<<<blocks, threads>>>(d_in, d_out, N);
        cudaDeviceSynchronize();
        int shared_max_res = 0;
        cudaMemcpy(&shared_max_res, d_out, sizeof(int), cudaMemcpyDeviceToHost);

        double speedup_cpu = cpu_ms / shared_ms;
        double speedup_glob = (double)glob_ms / shared_ms;

        printf("| 10^%-10d | %11.3f | %16.3f | %16.3f | %14.2fx | %13.2fx |\n",
               (int)log10(N), cpu_ms, glob_ms, shared_ms, speedup_cpu, speedup_glob);

        if (fp) {
            fprintf(fp, "%d,%.4f,%.4f,%.4f,%.3f,%.3f\n",
                    N, cpu_ms, glob_ms, shared_ms, speedup_cpu, speedup_glob);
        }

        // Verification check
        int diff_sum = abs(cpu_sum - shared_res);
        int diff_max = abs(cpu_max - shared_max_res);
        bool passed = (diff_sum == 0 && diff_max == 0);
        printf("  -> Verification (Sum Diff=%d, Max Diff=%d) [%s]\n",
               diff_sum, diff_max, passed ? "PASSED (100% Exact Match)" : "FAILED");

        cudaFree(d_in); cudaFree(d_out);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        free(h_in);
    }
    printf("+--------------+-------------+------------------+------------------+------------------+-----------------+\n");

    if (fp) fclose(fp);
    return 0;
}
