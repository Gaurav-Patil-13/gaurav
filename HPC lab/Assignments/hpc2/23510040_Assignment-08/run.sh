#!/bin/bash
# run.sh - Master Compilation and Benchmark Suite for Assignment 8

set -e

echo "================================================================="
echo "        ASSIGNMENT 8 - COMPILATION AND EXECUTION SUITE          "
echo "================================================================="

# Step 1: Collect System Configuration
echo "[STEP 1/5] Collecting System & MPI/OpenMP Environment Info..."
chmod +x sys_info.sh
./sys_info.sh

# Step 2: Compile C / OpenMP / MPI Source Files
echo ""
echo "[STEP 2/5] Compiling C Source Files (OpenMP & OpenMPI)..."

gcc -O2 task1a_matmul_seq.c -o task1a_matmul_seq
echo " -> Compiled task1a_matmul_seq"

gcc -O2 -fopenmp task1b_matmul_omp.c -o task1b_matmul_omp
echo " -> Compiled task1b_matmul_omp"

mpicc -O2 -fopenmp task1c_matmul_mpi.c -o task1c_matmul_mpi
echo " -> Compiled task1c_matmul_mpi"

gcc -O2 task2a_conv_seq.c -o task2a_conv_seq
echo " -> Compiled task2a_conv_seq"

gcc -O2 -fopenmp task2b_conv_omp.c -o task2b_conv_omp
echo " -> Compiled task2b_conv_omp"

mpicc -O2 -fopenmp task2c_conv_mpi.c -o task2c_conv_mpi
echo " -> Compiled task2c_conv_mpi"

mpicc -O2 -fopenmp task3_comparative_benchmark.c -o task3_comparative_benchmark
echo " -> Compiled task3_comparative_benchmark"

echo "All targets compiled successfully."

# Step 3: Run Benchmark Programs & Gather Data
echo ""
echo "[STEP 3/5] Executing Comparative Benchmarks..."

# Run Task 3 master benchmark suite (runs MatMul and Conv across sizes, cores, schedulers)
mpirun --allow-run-as-root -np 8 ./task3_comparative_benchmark > benchmark_results.txt 2>&1 || mpirun -np 8 ./task3_comparative_benchmark > benchmark_results.txt 2>&1

cat benchmark_results.txt

# Step 4: Generate Graphs using Matplotlib
echo ""
echo "[STEP 4/5] Generating Analysis Graphs..."
if command -v python3 &> /dev/null; then
    python3 generate_graphs.py
else
    python generate_graphs.py
fi

# Step 5: Create Word Document Reports (Report_8.docx and Report.docx)
echo ""
echo "[STEP 5/5] Generating DOCX Reports..."
if command -v python3 &> /dev/null; then
    python3 create_report8_docx.py
else
    python create_report8_docx.py
fi

echo ""
echo "================================================================="
echo "        ASSIGNMENT 8 EXECUTION & REPORT GENERATION COMPLETE      "
echo "================================================================="
