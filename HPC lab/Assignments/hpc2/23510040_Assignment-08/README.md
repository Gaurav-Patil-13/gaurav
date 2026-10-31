# Assignment 8: Comparative Study of OpenMP and MPI for Parallel Applications

## Overview
This directory contains the full implementation and benchmark suite for **Assignment 8: Comparative Study of OpenMP and MPI for Parallel Applications**.

The assignment compares **Shared-Memory Parallelism (OpenMP)** and **Distributed-Memory Message-Passing (MPI)** across two primary compute workloads:
1. **Task 1: Dense Matrix Multiplication ($N \times N$)**
2. **Task 2: 2D Image Convolution ($N \times N$ with $3 \times 3$ Kernel & Halo Exchange)**
3. **Task 3: Unified Comparative Benchmark & Automated Graph/Report Generator**

---

## Folder & File Structure
```
Assignment-8/
├── Experiment no 8.pdf            # Assignment Description & Specs
├── task1a_matmul_seq.c            # Sequential Matrix Multiplication
├── task1b_matmul_omp.c            # OpenMP Parallel Matrix Multiplication
├── task1c_matmul_mpi.c            # MPI Distributed Matrix Multiplication
├── task2a_conv_seq.c              # Sequential 2D Image Convolution
├── task2b_conv_omp.c              # OpenMP 2D Image Convolution (Static, Dynamic, Guided)
├── task2c_conv_mpi.c              # MPI 2D Image Convolution (with Ghost Row/Halo Exchange)
├── task3_comparative_benchmark.c  # Master Benchmarking Driver Suite
├── sys_info.sh                    # System Hardware & Software Info Collector
├── generate_graphs.py             # Python Matplotlib Script for Graphs 1-6
├── create_report8_docx.py         # Python Word Document Generator Script
├── run.sh                         # Automated Compilation, Execution & Benchmark Runner
├── README.md                      # Comprehensive Documentation
├── benchmark_results.txt          # Saved Output Logs (Generated after run)
├── system_configuration.txt       # System Specifications Log (Generated after run)
├── graphs/                        # Directory containing generated PNG graphs
│   ├── graph1_matmul_time_vs_cores.png
│   ├── graph2_matmul_speedup.png
│   ├── graph3_matmul_efficiency.png
│   ├── graph4_matmul_problem_size.png
│   ├── graph5_conv_time.png
│   └── graph6_omp_scheduling.png
├── Report_8.docx                  # Full Professional DOCX Report
└── Report.docx                    # Standard DOCX Report Copy
```

---

## Prerequisites
To compile and execute the programs, ensure the following are installed in your Linux/WSL environment:
- **GCC Compiler** with OpenMP support (`gcc -fopenmp`)
- **OpenMPI Library** (`mpicc`, `mpirun`)
- **Python 3** with `matplotlib`, `pandas`, `numpy`, and `python-docx`

In Ubuntu/WSL, install missing packages via:
```bash
sudo apt update
sudo apt install build-essential openmpi-bin libopenmpi-dev python3-matplotlib python3-pandas python3-docx
```

---

## How to Compile & Run

### Method 1: Automated Execution (Recommended)
Run the master bash script to compile all targets, execute benchmarks, collect system info, generate graphs, and generate Word reports:
```bash
chmod +x run.sh sys_info.sh
./run.sh
```

### Method 2: Manual Compilation & Execution

#### Task 1: Matrix Multiplication
```bash
# Sequential
gcc -O2 task1a_matmul_seq.c -o task1a_matmul_seq
./task1a_matmul_seq

# OpenMP (Shared Memory)
gcc -O2 -fopenmp task1b_matmul_omp.c -o task1b_matmul_omp
./task1b_matmul_omp

# MPI (Distributed Memory)
mpicc -O2 -fopenmp task1c_matmul_mpi.c -o task1c_matmul_mpi
mpirun -np 4 ./task1c_matmul_mpi 1000
```

#### Task 2: 2D Image Convolution
```bash
# Sequential
gcc -O2 task2a_conv_seq.c -o task2a_conv_seq
./task2a_conv_seq

# OpenMP
gcc -O2 -fopenmp task2b_conv_omp.c -o task2b_conv_omp
./task2b_conv_omp

# MPI with Halo Exchange
mpicc -O2 -fopenmp task2c_conv_mpi.c -o task2c_conv_mpi
mpirun -np 4 ./task2c_conv_mpi 2000
```

#### Task 3: Unified Benchmark Suite
```bash
mpicc -O2 -fopenmp task3_comparative_benchmark.c -o task3_comparative_benchmark
mpirun -np 8 ./task3_comparative_benchmark
```

---

## Key Features Implemented
1. **Matrix Multiplication (Task 1)**:
   - Dynamic memory allocation to handle $500 \times 500$, $1000 \times 1000$, and $2000 \times 2000$ matrices without stack overflow.
   - Cache-friendly loop ordering ($i, k, j$) and OpenMP `#pragma omp parallel for collapse(2)` directives.
   - Block row-wise decomposition in MPI using `MPI_Scatter`, `MPI_Bcast`, and `MPI_Gather`.

2. **2D Convolution (Task 2)**:
   - Support for multiple matrix problem sizes ($1000 \times 1000$, $2000 \times 2000$, $4000 \times 4000$).
   - Loop scheduling evaluations in OpenMP (`static`, `dynamic`, `guided`).
   - Ghost row / Halo exchange implementation in MPI using non-blocking/blocking neighbor exchanges (`MPI_Sendrecv`) to accurately process $3 \times 3$ kernel boundary rows across process subgrids.

3. **Benchmarking & Reporting (Task 3 & Scripts)**:
   - Automated metrics extraction (Speedup $S = T_{seq} / T_{par}$, Parallel Efficiency $E = S / p$).
   - Professional chart rendering saved as high-DPI PNGs in `graphs/`.
   - Comprehensive DOCX reports (`Report_8.docx` and `Report.docx`) matching the format of earlier assignments.
