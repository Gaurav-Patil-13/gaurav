#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vector_add_config_kernel(const float *A, const float *B, float *C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        C[id] = A[id] + B[id];
    }
}

int main() {
    const int N = 1000000; // 1 Million elements
    size_t bytes = N * sizeof(float);

    printf("=================================================================\n");
    printf(" ASSIGNMENT 9 - TASK 6: VECTOR ADDITION WITH DIFFERENT BLOCK SIZES \n");
    printf("=================================================================\n");
    printf("Fixed Vector Size N: %d elements\n\n", N);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.1f;
        h_B[i] = (float)(N - i) * 0.2f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int test_block_sizes[] = {64, 128, 256, 512};
    int num_tests = sizeof(test_block_sizes) / sizeof(test_block_sizes[0]);

    printf("Execution Benchmark Table:\n");
    printf("+-------------+---------------+---------------+--------------------+----------------+\n");
    printf("| Vector Size | Threads/Block | Blocks Needed | Total Threads Grid | Kernel Time (ms)|\n");
    printf("+-------------+---------------+---------------+--------------------+----------------+\n");

    // Open file to save CSV results for plotting Graph 2
    FILE *fp = fopen("task6_block_benchmark.csv", "w");
    if (fp) {
        fprintf(fp, "VectorSize,ThreadsPerBlock,BlocksNeeded,KernelTimeMs\n");
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int t = 0; t < num_tests; t++) {
        int threads_per_block = test_block_sizes[t];
        // Ceiling division: (N + block_size - 1) / block_size
        int blocks = (N + threads_per_block - 1) / threads_per_block;
        int total_threads = blocks * threads_per_block;

        // Warmup run
        vector_add_config_kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        // Timed run (average of 10 iterations)
        float total_ms = 0.0f;
        int num_runs = 10;
        for (int r = 0; r < num_runs; r++) {
            cudaEventRecord(start);
            vector_add_config_kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }
        float avg_ms = total_ms / num_runs;

        printf("| %-11d | %-13d | %-13d | %-18d | %-14.4f |\n",
               N, threads_per_block, blocks, total_threads, avg_ms);

        if (fp) {
            fprintf(fp, "%d,%d,%d,%.5f\n", N, threads_per_block, blocks, avg_ms);
        }
    }
    printf("+-------------+---------------+---------------+--------------------+----------------+\n");
    if (fp) fclose(fp);

    printf("\nAnalysis & Answers:\n");
    printf("1. Does increasing threads per block always improve performance?\n");
    printf("   -> NO. Beyond a certain threshold (e.g. 256 or 512), performance plateaus or can\n");
    printf("      even slightly degrade due to register pressure, reduced warp occupancy, and\n");
    printf("      limited shared memory per SM.\n");
    printf("2. What happens when the number of threads is not an exact multiple of N?\n");
    printf("   -> Ceiling division creates an extra block, causing the total thread count to exceed N.\n");
    printf("      The 'trailing' threads in the last block have Global IDs >= N.\n");
    printf("3. Why is the boundary check necessary?\n");
    printf("   -> Without 'if (id < N)', the trailing threads would write to memory addresses\n");
    printf("      outside the allocated array bounds, triggering segmentation faults, invalid\n");
    printf("      memory access errors (cudaErrorIllegalAddress), or data corruption.\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
