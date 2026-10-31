/**
 * ============================================================================
 * Task 3: Comprehensive Comparative Benchmark Suite
 * Experiment 11 - CUDA Performance Optimization & cuBLAS
 * ============================================================================
 * 
 * Compares:
 *   1. CPU Sequential Matrix Multiplication (O(N^3) baseline)
 *   2. Basic CUDA Matrix Multiplication (Global Memory)
 *   3. Tiled CUDA Matrix Multiplication (Shared Memory, Tile 8x8)
 *   4. Tiled CUDA Matrix Multiplication (Shared Memory, Tile 16x16)
 *   5. Tiled CUDA Matrix Multiplication (Shared Memory, Tile 32x32)
 *   6. cuBLAS SGEMM (Kernel Execution Time only)
 *   7. cuBLAS SGEMM (End-to-End: H2D Transfer + Kernel + D2H Transfer)
 * 
 * Matrix Dimensions:
 *   - N = 256, 512, 1024
 * 
 * Performance Metrics:
 *   - Execution Time (ms)
 *   - Speedup vs CPU
 *   - Speedup vs Basic CUDA
 *   - Computational Throughput (GFLOPS = 2*N^3 / (time * 1e9))
 *   - Memory Transfer Overhead vs Compute Time for cuBLAS
 * 
 * Exports CSV files for plotting and reporting:
 *   - matmul_comparison.csv
 *   - tile_size_comparison.csv
 *   - cublas_breakdown.csv
 * ============================================================================
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ \
                  << " - Status code: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ----------------------------------------------------------------------------
// 1. CPU Sequential Matrix Multiplication
// ----------------------------------------------------------------------------
void matmul_cpu(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ----------------------------------------------------------------------------
// 2. Basic CUDA Kernel (Global Memory)
// ----------------------------------------------------------------------------
__global__ void matmul_basic_kernel(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ----------------------------------------------------------------------------
// 3. Tiled CUDA Kernel (Shared Memory)
// ----------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C, int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int aRow = row;
        if (aRow < N && aCol < N) {
            s_A[threadIdx.y][threadIdx.x] = A[aRow * N + aCol];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < N && bCol < N) {
            s_B[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ----------------------------------------------------------------------------
// Helper: Numerical Verification
// ----------------------------------------------------------------------------
bool verify_matrices(const float *ref, const float *test, int N, float eps = 1e-2f) {
    for (int i = 0; i < N * N; ++i) {
        float diff = std::abs(ref[i] - test[i]);
        float maxVal = std::max(1.0f, std::max(std::abs(ref[i]), std::abs(test[i])));
        if (diff / maxVal > eps) {
            std::cerr << "Mismatch at index " << i << " (Row " << i / N << ", Col " << i % N
                      << "): Ref=" << ref[i] << ", Test=" << test[i] 
                      << ", RelDiff=" << (diff / maxVal) << std::endl;
            return false;
        }
    }
    return true;
}

// Struct to store results per matrix size
struct BenchmarkRecord {
    int N;
    float time_cpu_ms;
    float time_cuda_basic_ms;
    float time_cuda_tile8_ms;
    float time_cuda_tile16_ms;
    float time_cuda_tile32_ms;
    float time_cublas_kernel_ms;
    float time_cublas_total_ms;
    float time_h2d_ms;
    float time_d2h_ms;

    // Derived metrics
    double gflops_cpu;
    double gflops_cuda_basic;
    double gflops_cuda_tile16;
    double gflops_cublas_kernel;
    double gflops_cublas_total;
    float speedup_tile16_vs_cpu;
    float speedup_cublas_vs_cpu;
    float speedup_cublas_vs_basic;
    float speedup_cublas_vs_tile16;
};

int main(int argc, char **argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "      EXPERIMENT 11: CUDA PERFORMANCE OPTIMIZATION & cuBLAS BENCHMARK SUITE     " << std::endl;
    std::cout << "================================================================================" << std::endl;

    // GPU Device Properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "[Device Configuration]" << std::endl;
    std::cout << "  GPU Name:                  " << prop.name << std::endl;
    std::cout << "  Compute Capability:        " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Multiprocessors (SMs):     " << prop.multiProcessorCount << std::endl;
    std::cout << "  Total Global Memory:       " << (prop.totalGlobalMem / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Shared Memory per Block:   " << (prop.sharedMemPerBlock / 1024.0) << " KB" << std::endl;
    std::cout << "  Warp Size:                 " << prop.warpSize << std::endl;
    std::cout << "  Max Threads per Block:     " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    // Initialize cuBLAS
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    std::vector<int> test_sizes = {256, 512, 1024};
    std::vector<BenchmarkRecord> records;

    const int WARMUP_RUNS = 2;
    const int BENCH_RUNS = 5;

    for (int N : test_sizes) {
        std::cout << "\n>>> Benchmarking Matrix Dimension N = " << N << " (" << N << "x" << N 
                  << " Elements, Memory: " << (3 * N * N * sizeof(float) / (1024.0 * 1024.0)) 
                  << " MB for A,B,C) <<<" << std::endl;

        size_t bytes = N * N * sizeof(float);
        double total_flops = 2.0 * (double)N * (double)N * (double)N;

        // Allocate Host memory
        std::vector<float> h_A(N * N);
        std::vector<float> h_B(N * N);
        std::vector<float> h_C_cpu(N * N, 0.0f);
        std::vector<float> h_C_basic(N * N, 0.0f);
        std::vector<float> h_C_tile8(N * N, 0.0f);
        std::vector<float> h_C_tile16(N * N, 0.0f);
        std::vector<float> h_C_tile32(N * N, 0.0f);
        std::vector<float> h_C_cublas(N * N, 0.0f);

        // Deterministic pseudo-random initialization
        for (int i = 0; i < N * N; ++i) {
            h_A[i] = static_cast<float>(rand() % 100) / 100.0f;
            h_B[i] = static_cast<float>(rand() % 100) / 100.0f;
        }

        // Allocate Device memory
        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        CHECK_CUDA(cudaMalloc(&d_A, bytes));
        CHECK_CUDA(cudaMalloc(&d_B, bytes));
        CHECK_CUDA(cudaMalloc(&d_C, bytes));

        // Copy A and B to device
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

        BenchmarkRecord rec;
        rec.N = N;

        // --------------------------------------------------------------------
        // 1. CPU Sequential
        // --------------------------------------------------------------------
        std::cout << "  [1/6] Running CPU Sequential..." << std::flush;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        matmul_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        rec.time_cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
        rec.gflops_cpu = (total_flops / (rec.time_cpu_ms * 1.0e6));
        std::cout << " Done: " << std::fixed << std::setprecision(2) << rec.time_cpu_ms << " ms (" 
                  << rec.gflops_cpu << " GFLOPS)" << std::endl;

        // CUDA Timing Events
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // --------------------------------------------------------------------
        // 2. CUDA Basic (Global Memory)
        // --------------------------------------------------------------------
        std::cout << "  [2/6] Running CUDA Basic (Global Memory)..." << std::flush;
        dim3 blockBasic(16, 16);
        dim3 gridBasic((N + blockBasic.x - 1) / blockBasic.x, (N + blockBasic.y - 1) / blockBasic.y);

        // Warmup
        for (int w = 0; w < WARMUP_RUNS; ++w) {
            matmul_basic_kernel<<<gridBasic, blockBasic>>>(d_A, d_B, d_C, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < BENCH_RUNS; ++r) {
            matmul_basic_kernel<<<gridBasic, blockBasic>>>(d_A, d_B, d_C, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        rec.time_cuda_basic_ms = total_ms / BENCH_RUNS;
        rec.gflops_cuda_basic = total_flops / (rec.time_cuda_basic_ms * 1.0e6);

        CHECK_CUDA(cudaMemcpy(h_C_basic.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        bool basic_ok = verify_matrices(h_C_cpu.data(), h_C_basic.data(), N);
        std::cout << " Done: " << rec.time_cuda_basic_ms << " ms (" 
                  << rec.gflops_cuda_basic << " GFLOPS) [" << (basic_ok ? "VERIFIED" : "FAILED") << "]" << std::endl;

        // --------------------------------------------------------------------
        // 3. CUDA Tiled (Tile 8x8)
        // --------------------------------------------------------------------
        std::cout << "  [3a/6] Running CUDA Tiled (Shared Memory 8x8)..." << std::flush;
        dim3 block8(8, 8);
        dim3 grid8((N + 7) / 8, (N + 7) / 8);

        for (int w = 0; w < WARMUP_RUNS; ++w) {
            matmul_tiled_kernel<8><<<grid8, block8>>>(d_A, d_B, d_C, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < BENCH_RUNS; ++r) {
            matmul_tiled_kernel<8><<<grid8, block8>>>(d_A, d_B, d_C, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        rec.time_cuda_tile8_ms = total_ms / BENCH_RUNS;
        CHECK_CUDA(cudaMemcpy(h_C_tile8.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        bool tile8_ok = verify_matrices(h_C_cpu.data(), h_C_tile8.data(), N);
        std::cout << " Done: " << rec.time_cuda_tile8_ms << " ms [" << (tile8_ok ? "VERIFIED" : "FAILED") << "]" << std::endl;

        // --------------------------------------------------------------------
        // 4. CUDA Tiled (Tile 16x16)
        // --------------------------------------------------------------------
        std::cout << "  [3b/6] Running CUDA Tiled (Shared Memory 16x16)..." << std::flush;
        dim3 block16(16, 16);
        dim3 grid16((N + 15) / 16, (N + 15) / 16);

        for (int w = 0; w < WARMUP_RUNS; ++w) {
            matmul_tiled_kernel<16><<<grid16, block16>>>(d_A, d_B, d_C, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < BENCH_RUNS; ++r) {
            matmul_tiled_kernel<16><<<grid16, block16>>>(d_A, d_B, d_C, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        rec.time_cuda_tile16_ms = total_ms / BENCH_RUNS;
        rec.gflops_cuda_tile16 = total_flops / (rec.time_cuda_tile16_ms * 1.0e6);
        CHECK_CUDA(cudaMemcpy(h_C_tile16.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        bool tile16_ok = verify_matrices(h_C_cpu.data(), h_C_tile16.data(), N);
        std::cout << " Done: " << rec.time_cuda_tile16_ms << " ms (" 
                  << rec.gflops_cuda_tile16 << " GFLOPS) [" << (tile16_ok ? "VERIFIED" : "FAILED") << "]" << std::endl;

        // --------------------------------------------------------------------
        // 5. CUDA Tiled (Tile 32x32)
        // --------------------------------------------------------------------
        std::cout << "  [3c/6] Running CUDA Tiled (Shared Memory 32x32)..." << std::flush;
        dim3 block32(32, 32);
        dim3 grid32((N + 31) / 32, (N + 31) / 32);

        for (int w = 0; w < WARMUP_RUNS; ++w) {
            matmul_tiled_kernel<32><<<grid32, block32>>>(d_A, d_B, d_C, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < BENCH_RUNS; ++r) {
            matmul_tiled_kernel<32><<<grid32, block32>>>(d_A, d_B, d_C, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        rec.time_cuda_tile32_ms = total_ms / BENCH_RUNS;
        CHECK_CUDA(cudaMemcpy(h_C_tile32.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        bool tile32_ok = verify_matrices(h_C_cpu.data(), h_C_tile32.data(), N);
        std::cout << " Done: " << rec.time_cuda_tile32_ms << " ms [" << (tile32_ok ? "VERIFIED" : "FAILED") << "]" << std::endl;

        // --------------------------------------------------------------------
        // 6. cuBLAS (Kernel execution only)
        // --------------------------------------------------------------------
        std::cout << "  [4/6] Running cuBLAS sgemm (Kernel only)..." << std::flush;
        float alpha = 1.0f;
        float beta = 0.0f;

        // Warmup cuBLAS
        for (int w = 0; w < WARMUP_RUNS; ++w) {
            cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < BENCH_RUNS; ++r) {
            cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        rec.time_cublas_kernel_ms = total_ms / BENCH_RUNS;
        rec.gflops_cublas_kernel = total_flops / (rec.time_cublas_kernel_ms * 1.0e6);

        CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        bool cublas_ok = verify_matrices(h_C_cpu.data(), h_C_cublas.data(), N);
        std::cout << " Done: " << rec.time_cublas_kernel_ms << " ms (" 
                  << rec.gflops_cublas_kernel << " GFLOPS) [" << (cublas_ok ? "VERIFIED" : "FAILED") << "]" << std::endl;

        // --------------------------------------------------------------------
        // 7. cuBLAS End-to-End (H2D + Kernel + D2H)
        // --------------------------------------------------------------------
        std::cout << "  [5/6] Running cuBLAS End-to-End (Transfers + Kernel)..." << std::flush;
        
        // Measure H2D
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&rec.time_h2d_ms, start, stop));

        // Measure D2H
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&rec.time_d2h_ms, start, stop));

        rec.time_cublas_total_ms = rec.time_h2d_ms + rec.time_cublas_kernel_ms + rec.time_d2h_ms;
        rec.gflops_cublas_total = total_flops / (rec.time_cublas_total_ms * 1.0e6);
        std::cout << " Done: " << rec.time_cublas_total_ms << " ms (H2D: " << rec.time_h2d_ms 
                  << " ms, Kernel: " << rec.time_cublas_kernel_ms << " ms, D2H: " 
                  << rec.time_d2h_ms << " ms)" << std::endl;

        // Speedup calculations
        rec.speedup_tile16_vs_cpu = rec.time_cpu_ms / rec.time_cuda_tile16_ms;
        rec.speedup_cublas_vs_cpu = rec.time_cpu_ms / rec.time_cublas_kernel_ms;
        rec.speedup_cublas_vs_basic = rec.time_cuda_basic_ms / rec.time_cublas_kernel_ms;
        rec.speedup_cublas_vs_tile16 = rec.time_cuda_tile16_ms / rec.time_cublas_kernel_ms;

        records.push_back(rec);

        // Cleanup per size
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }

    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    // ------------------------------------------------------------------------
    // Print Summary Table to Console
    // ------------------------------------------------------------------------
    std::cout << "\n=========================================================================================================================" << std::endl;
    std::cout << "                                           COMPARATIVE BENCHMARK SUMMARY TABLE                                           " << std::endl;
    std::cout << "=========================================================================================================================" << std::endl;
    std::cout << std::left 
              << std::setw(6)  << "Size"
              << std::setw(14) << "CPU Seq (ms)"
              << std::setw(15) << "Basic CUDA (ms)"
              << std::setw(15) << "Tiled-16 (ms)"
              << std::setw(16) << "cuBLAS Ker (ms)"
              << std::setw(16) << "cuBLAS Tot (ms)"
              << std::setw(14) << "Spdup Tile/CPU"
              << std::setw(16) << "Spdup cuB/CPU"
              << std::setw(16) << "cuBLAS GFLOPS"
              << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------------------------------------" << std::endl;

    for (const auto &r : records) {
        std::cout << std::left 
                  << std::setw(6)  << r.N
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.time_cpu_ms
                  << std::setw(15) << r.time_cuda_basic_ms
                  << std::setw(15) << r.time_cuda_tile16_ms
                  << std::setw(16) << r.time_cublas_kernel_ms
                  << std::setw(16) << r.time_cublas_total_ms
                  << std::setw(14) << (std::to_string(static_cast<int>(r.speedup_tile16_vs_cpu)) + "x")
                  << std::setw(16) << (std::to_string(static_cast<int>(r.speedup_cublas_vs_cpu)) + "x")
                  << std::setw(16) << std::fixed << std::setprecision(1) << r.gflops_cublas_kernel
                  << std::endl;
    }
    std::cout << "=========================================================================================================================" << std::endl;

    // ------------------------------------------------------------------------
    // Export CSV 1: matmul_comparison.csv
    // ------------------------------------------------------------------------
    std::ofstream csv1("matmul_comparison.csv");
    if (csv1.is_open()) {
        csv1 << "MatrixSize,CPU_Time_ms,CUDABasic_Time_ms,CUDATiled_Time_ms,cuBLASKernel_Time_ms,cuBLASTotal_Time_ms,"
             << "CPU_GFLOPS,CUDABasic_GFLOPS,CUDATiled_GFLOPS,cuBLASKernel_GFLOPS,cuBLASTotal_GFLOPS,"
             << "Speedup_Tiled_vs_CPU,Speedup_cuBLAS_vs_CPU,Speedup_cuBLAS_vs_Basic,Speedup_cuBLAS_vs_Tiled\n";
        for (const auto &r : records) {
            csv1 << r.N << ","
                 << r.time_cpu_ms << ","
                 << r.time_cuda_basic_ms << ","
                 << r.time_cuda_tile16_ms << ","
                 << r.time_cublas_kernel_ms << ","
                 << r.time_cublas_total_ms << ","
                 << r.gflops_cpu << ","
                 << r.gflops_cuda_basic << ","
                 << r.gflops_cuda_tile16 << ","
                 << r.gflops_cublas_kernel << ","
                 << r.gflops_cublas_total << ","
                 << r.speedup_tile16_vs_cpu << ","
                 << r.speedup_cublas_vs_cpu << ","
                 << r.speedup_cublas_vs_basic << ","
                 << r.speedup_cublas_vs_tile16 << "\n";
        }
        csv1.close();
        std::cout << "Successfully exported: matmul_comparison.csv" << std::endl;
    }

    // ------------------------------------------------------------------------
    // Export CSV 2: tile_size_comparison.csv
    // ------------------------------------------------------------------------
    std::ofstream csv2("tile_size_comparison.csv");
    if (csv2.is_open()) {
        csv2 << "MatrixSize,Basic_Global_ms,Tile_8x8_ms,Tile_16x16_ms,Tile_32x32_ms,cuBLAS_ms\n";
        for (const auto &r : records) {
            csv2 << r.N << ","
                 << r.time_cuda_basic_ms << ","
                 << r.time_cuda_tile8_ms << ","
                 << r.time_cuda_tile16_ms << ","
                 << r.time_cuda_tile32_ms << ","
                 << r.time_cublas_kernel_ms << "\n";
        }
        csv2.close();
        std::cout << "Successfully exported: tile_size_comparison.csv" << std::endl;
    }

    // ------------------------------------------------------------------------
    // Export CSV 3: cublas_breakdown.csv
    // ------------------------------------------------------------------------
    std::ofstream csv3("cublas_breakdown.csv");
    if (csv3.is_open()) {
        csv3 << "MatrixSize,H2D_Transfer_ms,cuBLAS_Kernel_ms,D2H_Transfer_ms,Total_Time_ms,Compute_Percentage,Transfer_Percentage\n";
        for (const auto &r : records) {
            float compute_pct = (r.time_cublas_kernel_ms / r.time_cublas_total_ms) * 100.0f;
            float transfer_pct = ((r.time_h2d_ms + r.time_d2h_ms) / r.time_cublas_total_ms) * 100.0f;
            csv3 << r.N << ","
                 << r.time_h2d_ms << ","
                 << r.time_cublas_kernel_ms << ","
                 << r.time_d2h_ms << ","
                 << r.time_cublas_total_ms << ","
                 << compute_pct << ","
                 << transfer_pct << "\n";
        }
        csv3.close();
        std::cout << "Successfully exported: cublas_breakdown.csv" << std::endl;
    }

    std::cout << "\nTask 3 benchmark execution completed successfully." << std::endl;
    return 0;
}
