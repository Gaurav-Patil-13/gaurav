# Assignment 9: CUDA Fundamentals — GPU Architecture, Threads, Blocks, Grids and Basic Kernels

## Overview
This directory contains the complete source code, execution scripts, performance benchmark suites, visualization generators, and Word report documents for **Assignment 9: CUDA Fundamentals**.

The assignment investigates the **Single Instruction, Multiple Threads (SIMT)** execution paradigm on NVIDIA GPUs using the CUDA programming model. It covers:
1. **GPU Environment Discovery & Architectural Properties** (`task1_gpu_env.cu`)
2. **First CUDA Program: Hello from GPU across Block/Thread Configurations** (`task2_hello_gpu.cu`)
3. **Thread, Block, and Global ID Calculations** (`task3_thread_block_ids.cu`)
4. **Data-Parallel Vector Addition on GPU with Boundary Checks** (`task4_vector_add.cu`)
5. **CUDA Explicit Memory Management Pipeline (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)** (`task5_cuda_memory.cu`)
6. **Thread-Block Tuning and Occupancy Analysis ($64, 128, 256, 512$ threads/block)** (`task6_block_configurations.cu`)
7. **Performance Comparison: Sequential CPU vs OpenMP Multi-Core vs CUDA GPU** (`task7_comparative_vector_add.cu`)
8. **1D Array Processing: Element-wise Vector Multiplication & Array Squaring** (`task8_array_processing.cu`)

---

## Folder & File Structure
```
Assignment-9/
├── Experiment no 9.pdf            # Laboratory Assignment Problem Statement
├── task1_gpu_env.cu               # Task 1: Hardware GPU Properties & Environment Inspection
├── task2_hello_gpu.cu             # Task 2: CUDA Hello World Across 4 Block/Thread Configurations
├── task3_thread_block_ids.cu      # Task 3: Global Thread ID Computation Demonstration
├── task4_vector_add.cu            # Task 4: CPU vs GPU Vector Addition with Boundary Checks
├── task5_cuda_memory.cu           # Task 5: Complete CUDA Host-Device Memory Lifecycle
├── task6_block_configurations.cu  # Task 6: Block Size Parameter Tuning & Overhead Analysis
├── task7_comparative_vector_add.cu# Task 7: 3-Way Benchmark (Sequential vs OpenMP vs CUDA)
├── task8_array_processing.cu      # Task 8: 1D Array Processing (Element-wise Mult & Squaring)
├── sys_info.sh                    # System Hardware, NVIDIA GPU, and CUDA Compiler Info Collector
├── run.sh                         # Master Automated Compilation, Execution & Reporting Script
├── generate_graphs.py             # Python Matplotlib Script Generating High-DPI Analysis Graphs
├── create_report9_docx.py         # Word Document Report Generator
├── benchmark_results.txt          # Saved Benchmark Terminal Execution Logs
├── system_configuration.txt       # Hardware & CUDA Environment Log
├── graphs/                        # Directory Containing Generated High-DPI Visualizations
│   ├── graph1_input_size_vs_time.png
│   ├── graph2_block_size_vs_time.png
│   ├── graph3_input_size_vs_speedup.png
│   └── graph4_cuda_time_breakdown.png
├── Report_9.docx                  # Full Professional Word Laboratory Report
├── Report.docx                    # Standard Word Report Copy
└── README.md                      # Comprehensive Lab Documentation
```

---

## Prerequisites
To compile and execute the programs:
- **Operating System:** Linux / Ubuntu (WSL2 supported)
- **NVIDIA GPU:** CUDA-capable GPU (e.g., NVIDIA GeForce RTX 2050)
- **CUDA Toolkit:** Version 12.x or higher (`nvcc`)
- **Host C/C++ Compiler:** GCC/G++ with OpenMP support (`-fopenmp`)
- **Python 3:** With `matplotlib` and `python-docx`

Installation command for Ubuntu/WSL:
```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit gcc g++ python3-matplotlib python3-docx
```

---

## How to Compile & Run

### Automated Execution (Recommended)
Run the master bash script to compile all targets, execute benchmarks, generate graphs, and build the Word documents:
```bash
chmod +x run.sh sys_info.sh
./run.sh
```

### Manual Compilation & Execution
Each task can be compiled and run individually:

```bash
# Task 1: Explore GPU Environment
nvcc -O2 task1_gpu_env.cu -o task1_gpu_env
./task1_gpu_env

# Task 2: Hello from GPU
nvcc -O2 task2_hello_gpu.cu -o task2_hello_gpu
./task2_hello_gpu

# Task 3: Thread and Block IDs
nvcc -O2 task3_thread_block_ids.cu -o task3_thread_block_ids
./task3_thread_block_ids

# Task 4: CUDA Vector Addition
nvcc -O2 task4_vector_add.cu -o task4_vector_add
./task4_vector_add 1000000

# Task 5: CUDA Memory Management
nvcc -O2 task5_cuda_memory.cu -o task5_cuda_memory
./task5_cuda_memory

# Task 6: Block Configuration Tuning
nvcc -O2 task6_block_configurations.cu -o task6_block_configurations
./task6_block_configurations

# Task 7: Comparative Benchmark (Sequential vs OpenMP vs CUDA)
nvcc -O2 -Xcompiler -fopenmp task7_comparative_vector_add.cu -o task7_comparative_vector_add
./task7_comparative_vector_add

# Task 8: 1D Array Processing
nvcc -O2 task8_array_processing.cu -o task8_array_processing
./task8_array_processing 5000000
```

---

## Performance Summary & Key Insights
1. **Host-to-Device Transfer Overhead:** PCIe bus transfers ($H2D$ and $D2H$) dominate execution time for memory-bound vector addition.
2. **Pure Kernel Speedup:** When isolating GPU compute from PCIe transfer, the GPU achieves over $20\times$ speedup compared to sequential CPU code on large arrays ($10^7$ elements).
3. **Block Size Optimization:** Block sizes that are multiples of 32 (specifically 256 threads/block) achieve the highest hardware occupancy and lowest latency on the NVIDIA Ampere architecture.
4. **Boundary Checks:** The `if (id < N)` guard is essential to prevent illegal memory accesses for the trailing threads in ceiling-divided grid launches.
