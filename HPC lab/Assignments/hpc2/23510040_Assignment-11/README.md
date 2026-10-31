# Assignment 11: CUDA Performance Optimization and cuBLAS Library

## Overview
This directory contains the complete implementation, benchmarking suite, visualization scripts, and technical report documents for **Assignment 11: CUDA Performance Optimization and cuBLAS**.

The lab investigates GPU optimization strategies and industry-standard acceleration libraries:
1. **Custom CUDA Optimization:**
   - Naive matrix multiplication utilizing global memory (memory-bandwidth bound).
   - Tiled matrix multiplication utilizing on-chip `__shared__` memory to cache blocks of matrices $A$ and $B$, reducing global DRAM accesses by a factor of $\text{TILE}$.
   - Parameter sweep across tile dimensions ($8\times 8$, $16\times 16$, $32\times 32$) analyzing occupancy, thread block sizing, and register pressure.
2. **NVIDIA cuBLAS Acceleration:**
   - Utilizing NVIDIA's `cublasSgemm()` routine for Single-Precision General Matrix Multiply (SGEMM).
   - Seamless column-major to row-major mathematical mapping ($C_{col} = B_{col} \times A_{col} \iff C_{row} = A_{row} \times B_{row}$) with zero transpose overhead.
   - Host-to-Device and Device-to-Host memory transfer profiling vs pure compute kernel latency.
3. **Comparative Evaluation:**
   - Benchmarking across matrix sizes $N = 256, 512, 1024$.
   - Speedup metrics vs sequential CPU and naive CUDA implementations.
   - Effective computational throughput in GFLOPS and Roofline model analysis.

---

## Directory Structure
```
Assignment-11/
├── Experiment no 11.pdf            # Original Assignment Problem Statement
├── task1_matmul_custom.cu          # Task 1: Custom Naive & Tiled CUDA MatMul (Tile 8, 16, 32)
├── task2_matmul_cublas.cu          # Task 2: cuBLAS SGEMM Implementation & Memory Breakdown
├── task3_comparative_benchmark.cu  # Task 3: Comprehensive Comparative Benchmark Driver (CPU vs Custom vs cuBLAS)
├── sys_info.sh                     # System Hardware & CUDA/cuBLAS Environment Logger
├── run.sh                          # Master Automated Compilation, Execution & Benchmark Script
├── generate_graphs.py              # Matplotlib Script for Graphs 1 to 6
├── create_report11_docx.py         # Word Document Lab Report Generator
├── system_configuration.txt        # System Hardware & CUDA/cuBLAS Specifications Log
├── benchmark_results.txt           # Saved Benchmark Execution Logs
├── matmul_comparison.csv           # Master Comparative Benchmark Data
├── tile_size_comparison.csv        # Tile Size Sweep Data (8x8, 16x16, 32x32)
├── cublas_breakdown.csv            # cuBLAS Transfer vs Compute Breakdown
├── graphs/                         # High-DPI Performance Analysis Visualizations
│   ├── graph1_execution_time.png
│   ├── graph2_speedup_vs_cpu.png
│   ├── graph3_gflops_comparison.png
│   ├── graph4_tile_size_analysis.png
│   ├── graph5_cublas_breakdown.png
│   └── graph6_optimization_progression.png
├── Report_11.docx                  # Full Professional Word Laboratory Report
├── Report.docx                     # Standard Word Report Copy
└── README.md                       # Comprehensive Lab Documentation
```

---

## Prerequisites
- **Operating System:** Linux / Ubuntu (WSL2 supported)
- **NVIDIA GPU:** CUDA-capable GPU (tested on NVIDIA GeForce RTX 2050, 4 GB GDDR6 VRAM, Compute Capability 8.6)
- **CUDA Toolkit:** Version 12.x or higher (`nvcc`)
- **cuBLAS Library:** Included with CUDA Toolkit (`#include <cublas_v2.h>`, linked via `-lcublas`)
- **Host C/C++ Compiler:** GCC/G++ (`gcc`, `g++`)
- **Python 3:** With `matplotlib` and `python-docx` (`pip install matplotlib python-docx`)

---

## How to Compile & Run

### Automated Execution (Recommended)
Run the master script to collect hardware specs, compile all targets, execute benchmarks, plot high-resolution graphs, and compile docx reports:
```bash
chmod +x run.sh sys_info.sh
./run.sh
```

### Manual Compilation & Execution
Each task can also be compiled and executed independently using `nvcc -O2`:

```bash
# Task 1: Custom CUDA Matrix Multiplication (Naive vs Tiled 8x8, 16x16, 32x32)
nvcc -O2 task1_matmul_custom.cu -o task1_matmul_custom
./task1_matmul_custom

# Task 2: cuBLAS SGEMM Implementation
nvcc -O2 task2_matmul_cublas.cu -o task2_matmul_cublas -lcublas
./task2_matmul_cublas

# Task 3: Comprehensive Comparative Benchmark Driver
nvcc -O2 task3_comparative_benchmark.cu -o task3_comparative_benchmark -lcublas
./task3_comparative_benchmark

# Generate Visualizations
python3 generate_graphs.py

# Generate DOCX Reports
python3 create_report11_docx.py
```

---

## Summary of Optimization Layers

| Optimization Layer | Key Hardware Feature | Time ($N=1024$) | Speedup vs CPU | Throughput (GFLOPS) | Primary Bottleneck |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **CPU Sequential** | Host L1/L2/L3 Caches | $8470.12\text{ ms}$ | $1.0\times$ (Baseline) | $0.25\text{ GFLOPS}$ | Clock frequency, single thread |
| **Basic CUDA (Global)** | Thousands of CUDA Cores | $25.89\text{ ms}$ | $327.1\times$ | $82.94\text{ GFLOPS}$ | Off-chip DRAM latency ($200-800$ cycles) |
| **Tiled CUDA (Shared 8x8)** | On-Chip SRAM ($8\times$ reuse) | $7.59\text{ ms}$ | $1116.0\times$ | $282.94\text{ GFLOPS}$ | Low warp occupancy ($64\text{ threads/block}$) |
| **Tiled CUDA (Shared 16x16)**| On-Chip SRAM ($16\times$ reuse) | $5.52\text{ ms}$ | $\mathbf{1533.1\times}$ | $\mathbf{388.71\text{ GFLOPS}}$ | Balanced occupancy & shared memory bandwidth |
| **Tiled CUDA (Shared 32x32)**| On-Chip SRAM ($32\times$ reuse) | $6.10\text{ ms}$ | $1389.3\times$ | $352.26\text{ GFLOPS}$ | Max thread block limit ($1024\text{ threads/block}$) |
| **cuBLAS (Pure Kernel)** | Register Tiling + WMMA/Tensor Cores | $\mathbf{0.75\text{ ms}}$ | $\mathbf{11,266.1\times}$ | $\mathbf{2856.38\text{ GFLOPS}}$ | Hardware compute saturation ($\sim 2.85\text{ TFLOPS}$) |
| **cuBLAS (End-to-End)** | PCIe Bus Transfers + cuBLAS | $3.49\text{ ms}$ | $2428.7\times$ | $615.77\text{ GFLOPS}$ | Host-Device PCIe Transfer Latency ($78.4\%$ of time) |
