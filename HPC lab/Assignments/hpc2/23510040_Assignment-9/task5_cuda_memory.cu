#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void memory_demo_kernel(const float *d_A, const float *d_B, float *d_C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        d_C[id] = d_A[id] + d_B[id];
    }
}

int main() {
    int N = 500000;
    size_t bytes = N * sizeof(float);

    printf("=================================================================\n");
    printf("     ASSIGNMENT 9 - TASK 5: CUDA MEMORY MANAGEMENT PIPELINE      \n");
    printf("=================================================================\n");
    printf("Vector size: %d elements (%.2f MB)\n\n", N, bytes / (1024.0 * 1024.0));

    // [1] HOST ALLOCATION
    printf("[1] Host Memory Allocation (Host RAM):\n");
    printf("    -> Calling malloc() on CPU for h_A, h_B, h_C (Total: %.2f MB)\n", (3 * bytes) / (1024.0 * 1024.0));
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    // [2] DEVICE ALLOCATION
    printf("[2] Device Memory Allocation (GPU VRAM):\n");
    printf("    -> Calling cudaMalloc() to allocate device buffers d_A, d_B, d_C in GPU Global Memory\n");
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // [3] HOST TO DEVICE TRANSFER
    printf("[3] Data Transfer: Host to Device (H2D):\n");
    printf("    -> Calling cudaMemcpy(..., cudaMemcpyHostToDevice) over PCIe bus\n");
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // [4] KERNEL EXECUTION
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    printf("[4] Kernel Launch on Device:\n");
    printf("    -> Launching memory_demo_kernel<<<%d, %d>>>() across GPU Streaming Multiprocessors\n",
           blocks_per_grid, threads_per_block);
    memory_demo_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // [5] DEVICE TO HOST TRANSFER
    printf("[5] Data Transfer: Device to Host (D2H):\n");
    printf("    -> Calling cudaMemcpy(..., cudaMemcpyDeviceToHost) over PCIe bus\n");
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // [6] DEVICE DEALLOCATION
    printf("[6] Device Memory Deallocation:\n");
    printf("    -> Calling cudaFree() to release GPU Global Memory\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // [7] HOST DEALLOCATION
    printf("[7] Host Memory Deallocation:\n");
    printf("    -> Calling free() to release CPU RAM\n");
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\n=================================================================\n");
    printf(" CUDA MEMORY MANAGEMENT OPERATIONS SUMMARY TABLE                 \n");
    printf("=================================================================\n");
    printf(" Function       | Executing Entity | Memory Location Affected | Purpose\n");
    printf(" ---------------+------------------+--------------------------+--------------------------------------\n");
    printf(" malloc()       | Host (CPU)       | System RAM               | Allocates host virtual memory.\n");
    printf(" cudaMalloc()   | Host (CPU Driver)| GPU Global Memory (VRAM) | Allocates contiguous device memory.\n");
    printf(" cudaMemcpy(H2D)| Host/DMA Engine  | PCIe / Host to VRAM      | Copies input data to GPU memory.\n");
    printf(" Kernel Launch  | Device (GPU SMs) | GPU Register / L1 / L2   | Parallel arithmetic execution.\n");
    printf(" cudaMemcpy(D2H)| Host/DMA Engine  | PCIe / VRAM to Host      | Copies computed results back to CPU.\n");
    printf(" cudaFree()     | Host (CPU Driver)| GPU Global Memory (VRAM) | Releases allocated device memory.\n");
    printf(" free()         | Host (CPU)       | System RAM               | Releases CPU system memory.\n");

    return 0;
}
