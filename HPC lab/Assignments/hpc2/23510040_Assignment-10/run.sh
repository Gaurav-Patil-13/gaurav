#!/bin/bash
# run.sh - Master Compilation, Execution, Benchmark, and Report Generation for Assignment 10

set -e

echo "================================================================="
echo "        ASSIGNMENT 10 - CUDA OPTIMIZATION & ALGORITHMS SUITE     "
echo "================================================================="

# Step 1: Collect System and CUDA Hardware Specs
echo "[STEP 1/5] Collecting System & CUDA GPU Environment Info..."
chmod +x sys_info.sh
./sys_info.sh

# Step 2: Compile CUDA Source Files
echo ""
echo "[STEP 2/5] Compiling CUDA C++ Source Files using nvcc..."

nvcc -O2 task1_matmul_global.cu -o task1_matmul_global
echo " -> Compiled task1_matmul_global"

nvcc -O2 task2_matmul_shared.cu -o task2_matmul_shared
echo " -> Compiled task2_matmul_shared"

nvcc -O2 task3_matmul_compare.cu -o task3_matmul_compare
echo " -> Compiled task3_matmul_compare"

nvcc -O2 task4_reduction.cu -o task4_reduction
echo " -> Compiled task4_reduction"

nvcc -O2 task5_prefix_sum.cu -o task5_prefix_sum
echo " -> Compiled task5_prefix_sum"

nvcc -O2 task6_histogram.cu -o task6_histogram
echo " -> Compiled task6_histogram"

nvcc -O2 task7_benchmark_suite.cu -o task7_benchmark_suite
echo " -> Compiled task7_benchmark_suite"

echo "All 7 targets compiled successfully."

# Step 3: Run Programs and Log Benchmark Results
echo ""
echo "[STEP 3/5] Executing CUDA Programs and Benchmark Driver..."

OUTPUT_LOG="benchmark_results.txt"
> "$OUTPUT_LOG"

echo "Running Task 1 (Matrix Multiplication using Global Memory)..."
./task1_matmul_global | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 2 (Tiled Matrix Multiplication using Shared Memory)..."
./task2_matmul_shared | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 3 (Global Memory vs Shared Memory Direct Benchmark)..."
./task3_matmul_compare | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 4 (Parallel Reduction - Sum/Max/Min)..."
./task4_reduction | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 5 (Parallel Prefix Sum / Scan)..."
./task5_prefix_sum | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 6 (Parallel Histogram & Privatization)..."
./task6_histogram | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 7 (Comprehensive Comparative Benchmark Suite)..."
./task7_benchmark_suite | tee -a "$OUTPUT_LOG"

# Step 4: Generate Performance Graphs
echo ""
echo "[STEP 4/5] Generating Performance Analysis Graphs..."
if command -v python3 &> /dev/null; then
    python3 generate_graphs.py
else
    python generate_graphs.py
fi

# Step 5: Build DOCX Reports
echo ""
echo "[STEP 5/5] Generating Word Document Reports..."
if command -v python3 &> /dev/null; then
    python3 create_report10_docx.py
else
    python create_report10_docx.py
fi

echo ""
echo "================================================================="
echo "  ASSIGNMENT 10 EXECUTION, GRAPHS & REPORT GENERATION COMPLETE   "
echo "================================================================="
