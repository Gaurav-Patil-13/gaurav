#include <stdio.h>
#include <cuda_runtime.h>

__global__ void thread_block_id_kernel() {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int block_dim = blockDim.x;
    int global_id = block_id * block_dim + thread_id;

    printf("Block %d: Thread %d within Block -> Global Thread ID %d (blockDim = %d)\n",
           block_id, thread_id, global_id, block_dim);
}

int main() {
    printf("=================================================================\n");
    printf("    ASSIGNMENT 9 - TASK 3: UNDERSTAND THREAD AND BLOCK IDS       \n");
    printf("=================================================================\n");
    printf("Running configuration: 2 blocks x 4 threads per block (Total: 8)\n\n");

    printf("Expected Conceptual Structure:\n");
    printf("  Block 0:\n");
    printf("    Thread 0 -> Global ID 0\n");
    printf("    Thread 1 -> Global ID 1\n");
    printf("    Thread 2 -> Global ID 2\n");
    printf("    Thread 3 -> Global ID 3\n");
    printf("  Block 1:\n");
    printf("    Thread 0 -> Global ID 4\n");
    printf("    Thread 1 -> Global ID 5\n");
    printf("    Thread 2 -> Global ID 6\n");
    printf("    Thread 3 -> Global ID 7\n\n");

    printf("Actual GPU Kernel Output:\n");
    printf("-----------------------------------------------------------------\n");

    thread_block_id_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();

    printf("-----------------------------------------------------------------\n");
    printf("Mathematical Formula Used:\n");
    printf("  GlobalID = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("Where:\n");
    printf("  blockIdx.x  = ID of the thread block within the 1D grid\n");
    printf("  blockDim.x  = Number of threads contained in each block (4)\n");
    printf("  threadIdx.x = Local thread ID within the block (0 to 3)\n");

    return 0;
}
