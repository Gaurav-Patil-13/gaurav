#!/bin/bash
# run.sh - Master Compilation, Execution, Benchmark, and Report Generation for Assignment 11
# CUDA Performance Optimization and cuBLAS Library

set -e

# Ensure WSL2 NVIDIA driver library directory takes precedence over Linux stub libraries
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "================================================================="
echo "   ASSIGNMENT 11: CUDA PERFORMANCE OPTIMIZATION & cuBLAS SUITE   "
echo "================================================================="

# Step 1: Collect System and CUDA Hardware Specs
echo "[STEP 1/5] Collecting System, CUDA & cuBLAS Environment Info..."
chmod +x sys_info.sh
./sys_info.sh

# Step 2: Compile CUDA Source Files
echo ""
echo "[STEP 2/5] Compiling CUDA C++ Source Files using nvcc & cuBLAS..."

nvcc -O2 task1_matmul_custom.cu -o task1_matmul_custom
echo " -> Compiled task1_matmul_custom"

nvcc -O2 task2_matmul_cublas.cu -o task2_matmul_cublas -lcublas
echo " -> Compiled task2_matmul_cublas (-lcublas linked)"

nvcc -O2 task3_comparative_benchmark.cu -o task3_comparative_benchmark -lcublas
echo " -> Compiled task3_comparative_benchmark (-lcublas linked)"

echo "All 3 compilation targets built successfully."

# Step 3: Run Programs and Log Benchmark Results
echo ""
echo "[STEP 3/5] Executing CUDA Programs and Benchmark Driver..."

OUTPUT_LOG="benchmark_results.txt"
> "$OUTPUT_LOG"

echo "Running Task 1 (Custom CUDA Matrix Multiplication: Naive vs Tiled)..."
./task1_matmul_custom | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 2 (cuBLAS Matrix Multiplication & Layout Mapping)..."
./task2_matmul_cublas | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 3 (Comprehensive Comparative Benchmark Suite)..."
./task3_comparative_benchmark | tee -a "$OUTPUT_LOG"

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
    python3 create_report11_docx.py
else
    python create_report11_docx.py
fi

echo ""
echo "================================================================="
echo "  ASSIGNMENT 11 EXECUTION, GRAPHS & REPORT GENERATION COMPLETE   "
echo "================================================================="
