#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

// Global Memory Kernel
__global__ void matmul_global_k(const float *A, const float *B, float *C, int N) {
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

// Tiled Shared Memory Kernel
template <int TILE>
__global__ void matmul_shared_k(const float *A, const float *B, float *C, int N) {
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    int num_tiles = (N + TILE - 1) / TILE;

    for (int m = 0; m < num_tiles; m++) {
        if (row < N && (m * TILE + tx) < N) {
            s_A[ty][tx] = A[row * N + (m * TILE + tx)];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        if (col < N && (m * TILE + ty) < N) {
            s_B[ty][tx] = B[(m * TILE + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Sequential CPU
void matmul_cpu_ref(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

float time_global(const float *d_A, const float *d_B, float *d_C, int N, int bdim) {
    dim3 block(bdim, bdim);
    dim3 grid((N + bdim - 1) / bdim, (N + bdim - 1) / bdim);
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // Warm-up
    matmul_global_k<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    float total_ms = 0.0f;
    int runs = 5;
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(s);
        matmul_global_k<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, s, e);
        total_ms += ms;
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    return total_ms / runs;
}

template <int TILE>
float time_shared(const float *d_A, const float *d_B, float *d_C, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // Warm-up
    matmul_shared_k<TILE><<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    float total_ms = 0.0f;
    int runs = 5;
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(s);
        matmul_shared_k<TILE><<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, s, e);
        total_ms += ms;
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    return total_ms / runs;
}

int main() {
    printf("=================================================================\n");
    printf(" ASSIGNMENT 10 - TASK 3: GLOBAL MEMORY vs SHARED MEMORY TILED   \n");
    printf("=================================================================\n");

    cudaFree(0);

    FILE *fp = fopen("matmul_benchmark.csv", "w");
    if (fp) {
        fprintf(fp, "MatrixSize,BlockOrTile,CPUTimeMs,GlobalTimeMs,SharedTimeMs,SpeedupSharedOverGlobal,SpeedupOverCPU\n");
    }

    int test_sizes[] = {256, 512, 1024};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int tiles[] = {8, 16, 32};
    int num_tiles = sizeof(tiles) / sizeof(tiles[0]);

    printf("\nExecution Comparison Table:\n");
    printf("+-------------+------------+-------------+------------------+------------------+-------------------------+-----------------+\n");
    printf("| Matrix Size | Block/Tile | CPU Time(ms)| Global Mem Time  | Shared Mem Time  | Speedup (Shared/Global) | Speedup vs CPU  |\n");
    printf("+-------------+------------+-------------+------------------+------------------+-------------------------+-----------------+\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = test_sizes[s];
        size_t bytes = N * N * sizeof(float);

        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C = (float*)calloc(N * N, sizeof(float));

        for (int i = 0; i < N * N; i++) {
            h_A[i] = (float)(i % 50) * 0.02f;
            h_B[i] = (float)((i + 1) % 50) * 0.03f;
        }

        // CPU measurement
        auto t1 = std::chrono::high_resolution_clock::now();
        matmul_cpu_ref(h_A, h_B, h_C, N);
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, bytes);
        cudaMalloc((void**)&d_B, bytes);
        cudaMalloc((void**)&d_C, bytes);

        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        for (int t = 0; t < num_tiles; t++) {
            int tile = tiles[t];
            float g_time = time_global(d_A, d_B, d_C, N, tile);
            float s_time = 0.0f;
            if (tile == 8) s_time = time_shared<8>(d_A, d_B, d_C, N);
            else if (tile == 16) s_time = time_shared<16>(d_A, d_B, d_C, N);
            else if (tile == 32) s_time = time_shared<32>(d_A, d_B, d_C, N);

            double speedup_shared = (double)g_time / s_time;
            double speedup_cpu = cpu_ms / s_time;

            printf("| %4d x %-4d | %-10d | %11.2f | %13.3f ms | %13.3f ms | %21.2fx | %13.2fx |\n",
                   N, N, tile, cpu_ms, g_time, s_time, speedup_shared, speedup_cpu);

            if (fp) {
                fprintf(fp, "%d,%d,%.3f,%.4f,%.4f,%.3f,%.3f\n",
                        N, tile, cpu_ms, g_time, s_time, speedup_shared, speedup_cpu);
            }
        }
        printf("+-------------+------------+-------------+------------------+------------------+-------------------------+-----------------+\n");

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
    }

    if (fp) fclose(fp);

    printf("\nAnalytical Insights:\n");
    printf("1. Why does shared memory improve performance?\n");
    printf("   -> It reduces global memory DRAM accesses by a factor of TILE. Elements loaded\n");
    printf("      into fast on-chip SRAM are reused TILE times by all threads in the block.\n");
    printf("2. Does shared memory always provide the same improvement?\n");
    printf("   -> No, the speedup depends on tile size and arithmetic intensity. Small tiles (8)\n");
    printf("      have lower reuse, while TILE=16/32 yield maximum occupancy and bandwidth efficiency.\n");
    printf("3. What happens when the tile size changes?\n");
    printf("   -> TILE=8: Low occupancy and higher synchronization frequency.\n");
    printf("   -> TILE=16 (256 threads): Optimal occupancy and low register pressure.\n");
    printf("   -> TILE=32 (1024 threads): Maximum thread count per block, highest throughput on Ampere.\n");
    printf("4. What limitations does shared memory have?\n");
    printf("   -> Capacity is strictly limited (typically 48 KB - 100 KB per SM), and careless\n");
    printf("      stride access patterns can cause shared memory bank conflicts.\n");
    printf("5. How does data reuse affect performance?\n");
    printf("   -> In matrix multiplication, naive global accesses require 2*N memory loads per element.\n");
    printf("      Tiled shared memory reduces this to 2*N/TILE loads, shifting the kernel from\n");
    printf("      memory-bandwidth-bound closer to compute-bound.\n");

    return 0;
}
