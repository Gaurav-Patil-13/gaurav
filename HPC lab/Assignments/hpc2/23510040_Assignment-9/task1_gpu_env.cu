#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        return 1;
    }

    if (deviceCount == 0) {
        printf("There are no available CUDA-enabled device(s).\n");
        return 0;
    }

    printf("=================================================================\n");
    printf("         ASSIGNMENT 9 - TASK 1: EXPLORE GPU ENVIRONMENT         \n");
    printf("=================================================================\n");
    printf("Detected %d CUDA-capable device(s)\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        int driverVersion = 0, runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        printf("-----------------------------------------------------------------\n");
        printf("Device %d: \"%s\"\n", dev, prop.name);
        printf("-----------------------------------------------------------------\n");
        printf("  CUDA Driver Version / Runtime Version:         %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10,
               runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", prop.major, prop.minor);
        printf("  Total amount of global memory:                 %.2f MBytes (%.2f GBytes)\n",
               (float)prop.totalGlobalMem / 1048576.0f,
               (float)prop.totalGlobalMem / 1073741824.0f);
        printf("  Number of Streaming Multiprocessors (SMs):     %d\n", prop.multiProcessorCount);
        printf("  Total amount of constant memory:               %zu bytes\n", prop.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", prop.regsPerBlock);
        printf("  Warp size:                                     %d threads\n", prop.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", prop.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z):  (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z):  (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Memory Clock Rate:                             %.2f MHz\n", prop.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n", prop.memoryBusWidth);
        printf("  L2 Cache Size:                                 %d bytes (%.2f KB)\n",
               prop.l2CacheSize, (float)prop.l2CacheSize / 1024.0f);
        printf("\n");

        printf("Summary Parameters Table:\n");
        printf("  +----------------------------+--------------------------------------+\n");
        printf("  | Parameter                  | Value                                |\n");
        printf("  +----------------------------+--------------------------------------+\n");
        printf("  | GPU Name                   | %-36s |\n", prop.name);
        printf("  | GPU Memory                 | %.2f GB (%zu bytes)       |\n",
               (float)prop.totalGlobalMem / 1073741824.0f, prop.totalGlobalMem);
        printf("  | CUDA Runtime Version       | %d.%d                                  |\n",
               runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  | Compute Capability         | %d.%d                                  |\n", prop.major, prop.minor);
        printf("  | Number of SMs              | %-36d |\n", prop.multiProcessorCount);
        printf("  | Maximum Threads per Block  | %-36d |\n", prop.maxThreadsPerBlock);
        printf("  | Maximum Grid Dimensions    | (%d, %d, %d)           |\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  +----------------------------+--------------------------------------+\n");
    }

    printf("\nQuestions & Answers:\n");
    printf("1. What GPU is available on your system?\n");
    printf("   -> NVIDIA GeForce RTX 2050 (4 GB VRAM, Ampere Architecture)\n");
    printf("2. What is the maximum number of threads supported in one block?\n");
    printf("   -> 1024 threads per block.\n");
    printf("3. What is an SM?\n");
    printf("   -> A Streaming Multiprocessor (SM) is the fundamental hardware compute unit\n");
    printf("      of NVIDIA GPUs containing CUDA cores, tensor cores, registers, warp schedulers,\n");
    printf("      and shared memory/L1 cache.\n");
    printf("4. Why are GPU resources different from CPU cores?\n");
    printf("   -> CPUs prioritize low latency for sequential code using large caches and complex\n");
    printf("      out-of-order execution logic. GPUs prioritize high throughput for massive data\n");
    printf("      parallelism using thousands of smaller, energy-efficient SIMT cores.\n");

    return 0;
}
