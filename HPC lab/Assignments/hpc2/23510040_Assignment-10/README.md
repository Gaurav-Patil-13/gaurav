# Assignment 10: CUDA Optimization and Parallel Algorithms: Shared Memory, Reduction, Scan and Histogram

## Overview
This directory contains the complete implementation, automated benchmarking suite, visualization generators, and Word report documents for **Assignment 10: CUDA Optimization and Parallel Algorithms**.

The experiment investigates two complementary aspects of GPU programming:
1. **CUDA Memory Optimization:** Tiled matrix multiplication utilizing on-chip `__shared__` memory to reduce off-chip DRAM memory bandwidth bottlenecks.
2. **Fundamental Parallel Algorithms:** Designing and benchmarking foundational GPU primitives:
   - **Parallel Reduction:** Sum, Maximum, and Minimum via logarithmic shared-memory tree reduction ($\mathcal{O}(\log_2 N)$).
   - **Parallel Prefix Sum (Scan):** Inclusive scan using the Hillis–Steele shared-memory tree pattern with multi-block offset propagation.
   - **Parallel Histogram:** Resolving atomic memory serialization and contention via shared-memory privatization under uniform and skewed distributions.

---

## Folder & File Structure
```
Assignment-10/
├── Experiment 10.pdf           # Assignment Problem Statement
├── task1_matmul_global.cu      # Task 1: Matrix Multiplication using Global Memory
├── task2_matmul_shared.cu      # Task 2: Tiled Matrix Multiplication using Shared Memory
├── task3_matmul_compare.cu     # Task 3: Global Memory vs Shared Memory Direct Benchmark
├── task4_reduction.cu          # Task 4: Parallel Reduction (Sum, Max, Min)
├── task5_prefix_sum.cu         # Task 5: Parallel Prefix Sum / Scan (Hillis-Steele)
├── task6_histogram.cu          # Task 6: Parallel Histogram (Global Atomics vs Shared Privatization)
├── task7_benchmark_suite.cu    # Task 7: Comprehensive Comparative Driver (All 4 Algorithms)
├── sys_info.sh                 # Hardware & CUDA Environment Info Collector
├── run.sh                      # Master Automated Compilation, Execution & Benchmark Script
├── generate_graphs.py          # Python Matplotlib Script for Graphs 1 to 6
├── create_report10_docx.py     # Word Document Lab Report Generator
├── system_configuration.txt    # System Hardware & CUDA Specifications Log
├── benchmark_results.txt       # Saved Benchmark Execution Logs
├── matmul_benchmark.csv        # Matrix Multiplication Timing Data
├── reduction_benchmark.csv     # Reduction Timing Data
├── scan_benchmark.csv          # Prefix Sum Timing Data
├── histogram_benchmark.csv     # Histogram Timing Data
├── comparative_summary.csv     # Summary Comparative Data
├── graphs/                     # High-DPI Analysis Visualizations
│   ├── graph1_matmul_size_vs_time.png
│   ├── graph2_tile_size_vs_time.png
│   ├── graph3_reduction_size_vs_time.png
│   ├── graph4_scan_size_vs_time.png
│   ├── graph5_histogram_size_vs_time.png
│   └── graph6_cpu_vs_cuda_speedup.png
├── Report_10.docx              # Full Professional Word Laboratory Report
├── Report.docx                 # Standard Word Report Copy
└── README.md                   # Comprehensive Lab Documentation
```

---

## Prerequisites
- **Operating System:** Linux / Ubuntu (WSL2 supported)
- **NVIDIA GPU:** CUDA-capable GPU (tested on NVIDIA GeForce RTX 2050, 4 GB VRAM)
- **CUDA Toolkit:** Version 12.x or higher (`nvcc`)
- **Host C/C++ Compiler:** GCC/G++ (`gcc`, `g++`)
- **Python 3:** With `matplotlib` and `python-docx`

---

## How to Compile & Run

### Automated Execution (Recommended)
Execute the master runner to compile all code, perform benchmark runs across matrix dimensions, array sizes, and block configurations, generate all 6 graphs, and build the Word documents:
```bash
chmod +x run.sh sys_info.sh
./run.sh
```

### Manual Compilation & Execution
Each task can be compiled and executed independently using `nvcc -O2`:

```bash
# Task 1: Matrix Multiplication (Global Memory)
nvcc -O2 task1_matmul_global.cu -o task1_matmul_global
./task1_matmul_global

# Task 2: Tiled Matrix Multiplication (Shared Memory)
nvcc -O2 task2_matmul_shared.cu -o task2_matmul_shared
./task2_matmul_shared

# Task 3: Global vs Shared Memory Comparison
nvcc -O2 task3_matmul_compare.cu -o task3_matmul_compare
./task3_matmul_compare

# Task 4: Parallel Reduction (Sum/Max/Min)
nvcc -O2 task4_reduction.cu -o task4_reduction
./task4_reduction

# Task 5: Parallel Prefix Sum / Scan
nvcc -O2 task5_prefix_sum.cu -o task5_prefix_sum
./task5_prefix_sum

# Task 6: Parallel Histogram & Privatization
nvcc -O2 task6_histogram.cu -o task6_histogram
./task6_histogram

# Task 7: Comprehensive Comparative Suite
nvcc -O2 task7_benchmark_suite.cu -o task7_benchmark_suite
./task7_benchmark_suite
```

---

## Performance Highlights & Key Findings
1. **Shared Memory Tiling:** Reduces global memory traffic by a factor of $TILE$ (e.g. $16\times$ or $32\times$), transforming memory-bound matrix multiplication into high-throughput compute.
2. **Tree Reduction:** Shared memory reduction achieves $\mathcal{O}(\log_2 N)$ logarithmic time complexity, outperforming serialized global atomic accumulation by up to $30\times$.
3. **Histogram Privatization:** Resolves atomic memory contention under skewed data distributions by maintaining block-private histograms in fast SRAM, reducing atomic collisions by orders of magnitude.
4. **Prefix Sum:** Handles sequential dependencies across arbitrary array sizes using the Hillis–Steele tree scan pattern combined with multi-block offset distribution.
