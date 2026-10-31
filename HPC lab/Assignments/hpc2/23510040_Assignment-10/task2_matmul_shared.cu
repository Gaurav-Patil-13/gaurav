#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

// Tiled Matrix Multiplication Kernel using Shared Memory
template <int TILE>
__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    int num_tiles = (N + TILE - 1) / TILE;

    for (int m = 0; m < num_tiles; m++) {
        // Step 1 & 2: Load tiles into shared memory with boundary checks
        if (row < N && (m * TILE + tx) < N) {
            tileA[ty][tx] = A[row * N + (m * TILE + tx)];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        if (col < N && (m * TILE + ty) < N) {
            tileB[ty][tx] = B[(m * TILE + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        // Step 3: Synchronize to ensure all threads finish loading the tile
        __syncthreads();

        // Step 4: Compute partial dot product from shared memory
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        // Step 5: Synchronize before next tile overwrites shared memory
        __syncthreads();
    }

    // Write final accumulated sum to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Sequential CPU Baseline
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

template <int TILE>
void run_tiled_test(const float *d_A, const float *d_B, float *d_C, int N, double cpu_time) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    matmul_tiled_kernel<TILE><<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Timed run (average of 5 iterations)
    float total_ms = 0.0f;
    int iters = 5;
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        matmul_tiled_kernel<TILE><<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    float avg_ms = total_ms / iters;
    double speedup = cpu_time / avg_ms;
    double gflops = (2.0 * N * N * N) / (avg_ms * 1e-3 * 1e9);

    printf(" | TILE = %-2d    | %2dx%-11d | %3dx%-10d | %14.3f ms | %6.2fx | %7.2f |\n",
           TILE, block.x, block.y, grid.x, grid.y, avg_ms, speedup, gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=================================================================\n");
    printf(" ASSIGNMENT 10 - TASK 2: TILED MATRIX MULTIPLICATION (SHARED MEM)\n");
    printf("=================================================================\n");

    cudaFree(0);

    int test_sizes[] = {256, 512, 1024};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

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

        // Run CPU reference
        printf(" Calculating CPU sequential baseline...\n");
        auto t1 = std::chrono::high_resolution_clock::now();
        matmul_cpu(h_A, h_B, h_C_cpu, N);
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
        printf("  CPU Execution Time: %8.2f ms\n\n", cpu_time);

        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, bytes);
        cudaMalloc((void**)&d_B, bytes);
        cudaMalloc((void**)&d_C, bytes);

        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        printf(" Evaluating Shared Memory Tiling Configurations:\n");
        printf(" +--------------+---------------+---------------+-----------------+---------+---------+\n");
        printf(" | Tile Size    | Block Dim     | Grid Dim      | Kernel Time (ms)| Speedup | GFLOPS  |\n");
        printf(" +--------------+---------------+---------------+-----------------+---------+---------+\n");

        run_tiled_test<8>(d_A, d_B, d_C, N, cpu_time);
        run_tiled_test<16>(d_A, d_B, d_C, N, cpu_time);
        run_tiled_test<32>(d_A, d_B, d_C, N, cpu_time);

        printf(" +--------------+---------------+---------------+-----------------+---------+---------+\n");

        // Verify with TILE=16 output
        cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
        float max_diff = 0.0f;
        for (int i = 0; i < N * N; i++) {
            float diff = fabsf(h_C_cpu[i] - h_C_gpu[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf(" Verification: Max Difference = %e [%s]\n",
               max_diff, (max_diff < 1e-3f) ? "PASSED" : "FAILED");

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    }

    return 0;
}
