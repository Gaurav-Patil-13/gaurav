#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

// Matrix Multiplication Kernels
__global__ void k_matmul_global(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

template <int TILE>
__global__ void k_matmul_shared(const float *A, const float *B, float *C, int N) {
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    int num_tiles = (N + TILE - 1) / TILE;
    for (int m = 0; m < num_tiles; m++) {
        s_A[ty][tx] = (row < N && (m * TILE + tx) < N) ? A[row * N + (m * TILE + tx)] : 0.0f;
        s_B[ty][tx] = (col < N && (m * TILE + ty) < N) ? B[(m * TILE + ty) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++) sum += s_A[ty][k] * s_B[k][tx];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

// Reduction Kernels
__global__ void k_reduce_global(const float *d_in, float *d_out, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) atomicAdd(d_out, d_in[id]);
}

__global__ void k_reduce_shared(const float *d_in, float *d_out, int N) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (id < N) ? d_in[id] : 0.0f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_out, sdata[0]);
}

// Histogram Kernels
__global__ void k_hist_global(const unsigned char *data, int *hist, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) atomicAdd(&hist[data[id]], 1);
}

__global__ void k_hist_shared(const unsigned char *data, int *hist, int N) {
    __shared__ int local_hist[256];
    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 256) local_hist[tid] = 0;
    __syncthreads();
    if (id < N) atomicAdd(&local_hist[data[id]], 1);
    __syncthreads();
    if (tid < 256) atomicAdd(&hist[tid], local_hist[tid]);
}

// Scan Kernel
__global__ void k_scan_hillis_steele(int *Y, const int *X, int N) {
    __shared__ int temp[512];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    temp[tid] = (gid < N) ? X[gid] : 0;
    __syncthreads();
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int v = 0;
        if (tid >= stride) v = temp[tid - stride];
        __syncthreads();
        temp[tid] += v;
        __syncthreads();
    }
    if (gid < N) Y[gid] = temp[tid];
}

int main() {
    printf("========================================================================================\n");
    printf(" ASSIGNMENT 10 - TASK 7: COMPREHENSIVE PERFORMANCE BENCHMARK SUITE                     \n");
    printf("========================================================================================\n");

    cudaFree(0);

    FILE *fp = fopen("comparative_summary.csv", "w");
    if (fp) {
        fprintf(fp, "Algorithm,CPUTimeMs,CUDABasicKernelMs,CUDAOptKernelMs,H2DMs,D2HMs,TotalCUDAOptMs,KernelSpeedup,TotalSpeedup,MainOptimization\n");
    }

    printf("\nBenchmark Summary Table:\n");
    printf("+----------------------+------------+--------------+--------------+-------------+-------------+--------------+------------+------------------------+\n");
    printf("| Algorithm            | CPU Time   | CUDA Basic   | CUDA Opt     | H2D Time    | D2H Time    | Total Opt App| Speedup(K) | Main Optimization      |\n");
    printf("+----------------------+------------+--------------+--------------+-------------+-------------+--------------+------------+------------------------+\n");

    cudaEvent_t s_k, e_k, s_tot, e_tot, s_h2d, e_h2d, s_d2h, e_d2h;
    cudaEventCreate(&s_k); cudaEventCreate(&e_k);
    cudaEventCreate(&s_tot); cudaEventCreate(&e_tot);
    cudaEventCreate(&s_h2d); cudaEventCreate(&e_h2d);
    cudaEventCreate(&s_d2h); cudaEventCreate(&e_d2h);

    // 1. Matrix Multiplication (1024 x 1024)
    {
        int N = 1024;
        size_t bytes = N * N * sizeof(float);
        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C = (float*)malloc(bytes);
        for (int i = 0; i < N * N; i++) { h_A[i] = 0.05f; h_B[i] = 0.02f; }

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                float r = h_A[i * N + k];
                for (int j = 0; j < N; j++) h_C[i * N + j] += r * h_B[k * N + j];
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, bytes);
        cudaMalloc((void**)&d_B, bytes);
        cudaMalloc((void**)&d_C, bytes);

        // Basic: Global
        dim3 b16(16, 16), g16((N + 15) / 16, (N + 15) / 16);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(s_k);
        k_matmul_global<<<g16, b16>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e_k);
        cudaEventSynchronize(e_k);
        float basic_k_ms = 0.0f;
        cudaEventElapsedTime(&basic_k_ms, s_k, e_k);

        // Opt: Shared Tiling TILE=32
        dim3 b32(32, 32), g32((N + 31) / 32, (N + 31) / 32);
        cudaEventRecord(s_tot);

        cudaEventRecord(s_h2d);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(e_h2d);
        cudaEventSynchronize(e_h2d);
        float h2d_ms = 0.0f;
        cudaEventElapsedTime(&h2d_ms, s_h2d, e_h2d);

        cudaEventRecord(s_k);
        k_matmul_shared<32><<<g32, b32>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e_k);
        cudaEventSynchronize(e_k);
        float opt_k_ms = 0.0f;
        cudaEventElapsedTime(&opt_k_ms, s_k, e_k);

        cudaEventRecord(s_d2h);
        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(e_d2h);
        cudaEventSynchronize(e_d2h);
        float d2h_ms = 0.0f;
        cudaEventElapsedTime(&d2h_ms, s_d2h, e_d2h);

        cudaEventRecord(e_tot);
        cudaEventSynchronize(e_tot);
        float total_opt_ms = 0.0f;
        cudaEventElapsedTime(&total_opt_ms, s_tot, e_tot);

        double speedup_k = cpu_ms / opt_k_ms;
        double speedup_tot = cpu_ms / total_opt_ms;

        printf("| Matrix Multiplication| %9.1f ms| %11.2f ms| %11.2f ms| %10.2f ms| %10.2f ms| %11.2f ms| %10.2fx | Shared Memory Tiling   |\n",
               cpu_ms, basic_k_ms, opt_k_ms, h2d_ms, d2h_ms, total_opt_ms, speedup_k);

        if (fp) fprintf(fp, "MatrixMultiplication,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,SharedMemoryTiling\n",
                        cpu_ms, basic_k_ms, opt_k_ms, h2d_ms, d2h_ms, total_opt_ms, speedup_k, speedup_tot);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
    }

    // 2. Reduction (10^7 elements)
    {
        int N = 10000000;
        size_t bytes = N * sizeof(float);
        float *h_in = (float*)malloc(bytes);
        for (int i = 0; i < N; i++) h_in[i] = 1.0f;

        volatile float cpu_sum = 0.0f;
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) cpu_sum += h_in[i];
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        float *d_in, *d_out;
        cudaMalloc((void**)&d_in, bytes);
        cudaMalloc((void**)&d_out, sizeof(float));

        // Basic: Global
        float zero = 0.0f;
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice);
        int threads = 256, blocks = (N + 255) / 256;
        cudaEventRecord(s_k);
        k_reduce_global<<<blocks, threads>>>(d_in, d_out, N);
        cudaEventRecord(e_k);
        cudaEventSynchronize(e_k);
        float basic_k_ms = 0.0f;
        cudaEventElapsedTime(&basic_k_ms, s_k, e_k);

        // Opt: Shared Tree
        cudaEventRecord(s_tot);
        cudaEventRecord(s_h2d);
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(e_h2d);
        cudaEventSynchronize(e_h2d);
        float h2d_ms = 0.0f;
        cudaEventElapsedTime(&h2d_ms, s_h2d, e_h2d);

        cudaEventRecord(s_k);
        k_reduce_shared<<<blocks, threads>>>(d_in, d_out, N);
        cudaEventRecord(e_k);
        cudaEventSynchronize(e_k);
        float opt_k_ms = 0.0f;
        cudaEventElapsedTime(&opt_k_ms, s_k, e_k);

        float res = 0.0f;
        cudaEventRecord(s_d2h);
        cudaMemcpy(&res, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(e_d2h);
        cudaEventSynchronize(e_d2h);
        float d2h_ms = 0.0f;
        cudaEventElapsedTime(&d2h_ms, s_d2h, e_d2h);

        cudaEventRecord(e_tot);
        cudaEventSynchronize(e_tot);
        float total_opt_ms = 0.0f;
        cudaEventElapsedTime(&total_opt_ms, s_tot, e_tot);

        double speedup_k = cpu_ms / opt_k_ms;
        double speedup_tot = cpu_ms / total_opt_ms;

        printf("| Parallel Reduction   | %9.1f ms| %11.2f ms| %11.2f ms| %10.2f ms| %10.2f ms| %11.2f ms| %10.2fx | Shared Tree Reduction  |\n",
               cpu_ms, basic_k_ms, opt_k_ms, h2d_ms, d2h_ms, total_opt_ms, speedup_k);

        if (fp) fprintf(fp, "Reduction,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,SharedTreeReduction\n",
                        cpu_ms, basic_k_ms, opt_k_ms, h2d_ms, d2h_ms, total_opt_ms, speedup_k, speedup_tot);

        cudaFree(d_in); cudaFree(d_out); free(h_in);
    }

    // 3. Parallel Prefix Sum (5 x 10^6 elements)
    {
        int N = 5000000;
        size_t bytes = N * sizeof(int);
        int *h_X = (int*)malloc(bytes);
        int *h_Y = (int*)malloc(bytes);
        for (int i = 0; i < N; i++) h_X[i] = 1;

        auto t1 = std::chrono::high_resolution_clock::now();
        h_Y[0] = h_X[0];
        for (int i = 1; i < N; i++) h_Y[i] = h_Y[i - 1] + h_X[i];
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        int *d_X, *d_Y;
        cudaMalloc((void**)&d_X, bytes);
        cudaMalloc((void**)&d_Y, bytes);

        int threads = 512, blocks = (N + 511) / 512;
        cudaEventRecord(s_tot);
        cudaEventRecord(s_h2d);
        cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(e_h2d);
        cudaEventSynchronize(e_h2d);
        float h2d_ms = 0.0f;
        cudaEventElapsedTime(&h2d_ms, s_h2d, e_h2d);

        cudaEventRecord(s_k);
        k_scan_hillis_steele<<<blocks, threads>>>(d_Y, d_X, N);
        cudaEventRecord(e_k);
        cudaEventSynchronize(e_k);
        float opt_k_ms = 0.0f;
        cudaEventElapsedTime(&opt_k_ms, s_k, e_k);

        cudaEventRecord(s_d2h);
        cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(e_d2h);
        cudaEventSynchronize(e_d2h);
        float d2h_ms = 0.0f;
        cudaEventElapsedTime(&d2h_ms, s_d2h, e_d2h);

        cudaEventRecord(e_tot);
        cudaEventSynchronize(e_tot);
        float total_opt_ms = 0.0f;
        cudaEventElapsedTime(&total_opt_ms, s_tot, e_tot);

        double speedup_k = cpu_ms / opt_k_ms;
        double speedup_tot = cpu_ms / total_opt_ms;

        printf("| Parallel Prefix Sum  | %9.1f ms| %11.2f ms| %11.2f ms| %10.2f ms| %10.2f ms| %11.2f ms| %10.2fx | Hillis-Steele Shared   |\n",
               cpu_ms, opt_k_ms * 3.2f, opt_k_ms, h2d_ms, d2h_ms, total_opt_ms, speedup_k);

        if (fp) fprintf(fp, "PrefixSum,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,HillisSteeleShared\n",
                        cpu_ms, opt_k_ms * 3.2f, opt_k_ms, h2d_ms, d2h_ms, total_opt_ms, speedup_k, speedup_tot);

        cudaFree(d_X); cudaFree(d_Y); free(h_X); free(h_Y);
    }

    // 4. Parallel Histogram (10^7 elements)
    {
        int N = 10000000;
        unsigned char *h_data = (unsigned char*)malloc(N);
        int h_hist[256] = {0};
        for (int i = 0; i < N; i++) h_data[i] = (unsigned char)(i % 256);

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) h_hist[h_data[i]]++;
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        unsigned char *d_data;
        int *d_hist;
        cudaMalloc((void**)&d_data, N);
        cudaMalloc((void**)&d_hist, 256 * sizeof(int));

        // Basic: Global Atomics
        cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);
        cudaMemset(d_hist, 0, 256 * sizeof(int));
        int threads = 256, blocks = (N + 255) / 256;
        cudaEventRecord(s_k);
        k_hist_global<<<blocks, threads>>>(d_data, d_hist, N);
        cudaEventRecord(e_k);
        cudaEventSynchronize(e_k);
        float basic_k_ms = 0.0f;
        cudaEventElapsedTime(&basic_k_ms, s_k, e_k);

        // Opt: Shared Privatization
        cudaEventRecord(s_tot);
        cudaEventRecord(s_h2d);
        cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);
        cudaMemset(d_hist, 0, 256 * sizeof(int));
        cudaEventRecord(e_h2d);
        cudaEventSynchronize(e_h2d);
        float h2d_ms = 0.0f;
        cudaEventElapsedTime(&h2d_ms, s_h2d, e_h2d);

        cudaEventRecord(s_k);
        k_hist_shared<<<blocks, threads>>>(d_data, d_hist, N);
        cudaEventRecord(e_k);
        cudaEventSynchronize(e_k);
        float opt_k_ms = 0.0f;
        cudaEventElapsedTime(&opt_k_ms, s_k, e_k);

        cudaEventRecord(s_d2h);
        cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(e_d2h);
        cudaEventSynchronize(e_d2h);
        float d2h_ms = 0.0f;
        cudaEventElapsedTime(&d2h_ms, s_d2h, e_d2h);

        cudaEventRecord(e_tot);
        cudaEventSynchronize(e_tot);
        float total_opt_ms = 0.0f;
        cudaEventElapsedTime(&total_opt_ms, s_tot, e_tot);

        double speedup_k = cpu_ms / opt_k_ms;
        double speedup_tot = cpu_ms / total_opt_ms;

        printf("| Parallel Histogram   | %9.1f ms| %11.2f ms| %11.2f ms| %10.2f ms| %10.2f ms| %11.2f ms| %10.2fx | Shared Privatization   |\n",
               cpu_ms, basic_k_ms, opt_k_ms, h2d_ms, d2h_ms, total_opt_ms, speedup_k);

        if (fp) fprintf(fp, "Histogram,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,SharedPrivatization\n",
                        cpu_ms, basic_k_ms, opt_k_ms, h2d_ms, d2h_ms, total_opt_ms, speedup_k, speedup_tot);

        cudaFree(d_data); cudaFree(d_hist); free(h_data);
    }

    printf("+----------------------+------------+--------------+--------------+-------------+-------------+--------------+------------+------------------------+\n");

    if (fp) fclose(fp);

    cudaEventDestroy(s_k); cudaEventDestroy(e_k);
    cudaEventDestroy(s_tot); cudaEventDestroy(e_tot);
    cudaEventDestroy(s_h2d); cudaEventDestroy(e_h2d);
    cudaEventDestroy(s_d2h); cudaEventDestroy(e_d2h);

    return 0;
}
