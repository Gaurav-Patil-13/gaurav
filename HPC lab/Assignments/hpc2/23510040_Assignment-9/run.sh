#!/bin/bash
# run.sh - Master Compilation, Execution, Benchmark, and Report Generation for Assignment 9

set -e

echo "================================================================="
echo "        ASSIGNMENT 9 - CUDA COMPILATION AND BENCHMARK SUITE      "
echo "================================================================="

# Step 1: System and CUDA Hardware Collection
echo "[STEP 1/5] Collecting System & CUDA GPU Environment Info..."
chmod +x sys_info.sh
./sys_info.sh

# Step 2: Compile CUDA Source Files with nvcc
echo ""
echo "[STEP 2/5] Compiling CUDA C++ Source Files using nvcc..."

nvcc -O2 task1_gpu_env.cu -o task1_gpu_env
echo " -> Compiled task1_gpu_env"

nvcc -O2 task2_hello_gpu.cu -o task2_hello_gpu
echo " -> Compiled task2_hello_gpu"

nvcc -O2 task3_thread_block_ids.cu -o task3_thread_block_ids
echo " -> Compiled task3_thread_block_ids"

nvcc -O2 task4_vector_add.cu -o task4_vector_add
echo " -> Compiled task4_vector_add"

nvcc -O2 task5_cuda_memory.cu -o task5_cuda_memory
echo " -> Compiled task5_cuda_memory"

nvcc -O2 task6_block_configurations.cu -o task6_block_configurations
echo " -> Compiled task6_block_configurations"

nvcc -O2 -Xcompiler -fopenmp task7_comparative_vector_add.cu -o task7_comparative_vector_add
echo " -> Compiled task7_comparative_vector_add"

nvcc -O2 task8_array_processing.cu -o task8_array_processing
echo " -> Compiled task8_array_processing"

echo "All 8 CUDA targets compiled successfully."

# Step 3: Run Programs and Record Outputs
echo ""
echo "[STEP 3/5] Executing CUDA Programs and Benchmarks..."

OUTPUT_LOG="benchmark_results.txt"
> "$OUTPUT_LOG"

echo "Running Task 1 (Explore GPU Environment)..."
./task1_gpu_env | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 2 (Hello from GPU)..."
./task2_hello_gpu | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 3 (Thread and Block IDs)..."
./task3_thread_block_ids | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 4 (Vector Addition Using CUDA)..."
./task4_vector_add 1000000 | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 5 (CUDA Memory Management Pipeline)..."
./task5_cuda_memory | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 6 (Block Configurations Benchmark)..."
./task6_block_configurations | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 7 (Comparative Benchmark: Sequential vs OpenMP vs CUDA)..."
./task7_comparative_vector_add | tee -a "$OUTPUT_LOG"

echo "" >> "$OUTPUT_LOG"
echo "Running Task 8 (1D Array Processing: Element-wise Mult & Square)..."
./task8_array_processing 5000000 | tee -a "$OUTPUT_LOG"

# Step 4: Generate Graphs
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
    python3 create_report9_docx.py
else
    python create_report9_docx.py
fi

echo ""
echo "================================================================="
echo "   ASSIGNMENT 9 EXECUTION, GRAPHS & REPORT GENERATION COMPLETE   "
echo "================================================================="
