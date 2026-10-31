#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_gpu_kernel() {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU Thread %d (Block %d, Local Thread %d)\n", global_tid, bid, tid);
}

void run_configuration(int num_blocks, int threads_per_block) {
    int total_threads = num_blocks * threads_per_block;
    printf("\n=================================================================\n");
    printf(" Launching Kernel: %d block(s), %d threads/block (Total: %d threads)\n",
           num_blocks, threads_per_block, total_threads);
    printf("=================================================================\n");

    hello_gpu_kernel<<<num_blocks, threads_per_block>>>();
    cudaDeviceSynchronize(); // Ensure GPU printf flushes before CPU continues

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

int main() {
    printf("=================================================================\n");
    printf("   ASSIGNMENT 9 - TASK 2: FIRST CUDA PROGRAM (HELLO FROM GPU)    \n");
    printf("=================================================================\n");

    // Config 1: 1 block, 8 threads
    run_configuration(1, 8);

    // Config 2: 1 block, 16 threads
    run_configuration(1, 16);

    // Config 3: 2 blocks, 8 threads per block
    run_configuration(2, 8);

    // Config 4: 4 blocks, 8 threads per block
    run_configuration(4, 8);

    printf("\n=================================================================\n");
    printf(" OBSERVATION & ANALYSIS TABLE                                    \n");
    printf("=================================================================\n");
    printf(" Blocks | Threads/Block | Total Threads | Output Observation\n");
    printf("   1    |       8       |       8       | Threads 0 to 7 execute in single block 0.\n");
    printf("   1    |      16       |      16       | Threads 0 to 15 execute in single block 0.\n");
    printf("   2    |       8       |      16       | Block 0 has threads 0-7; Block 1 has threads 0-7.\n");
    printf("   4    |       8       |      32       | 4 blocks execute concurrently with 8 threads each.\n");
    printf("\nThink Answer:\n");
    printf("Does changing the number of blocks change the thread ID (threadIdx.x) values?\n");
    printf("-> NO. threadIdx.x is local to each block and always ranges from 0 to (blockDim.x - 1).\n");
    printf("   Adding more blocks creates additional block instances (different blockIdx.x),\n");
    printf("   but within each block, threadIdx.x restarts from 0.\n");
    printf("   To obtain a unique identifier across the whole grid, Global Thread ID must be computed:\n");
    printf("   GlobalID = blockIdx.x * blockDim.x + threadIdx.x\n");

    return 0;
}
