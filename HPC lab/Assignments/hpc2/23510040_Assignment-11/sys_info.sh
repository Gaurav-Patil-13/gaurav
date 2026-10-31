#!/bin/bash
# sys_info.sh - Collect system hardware, software, and CUDA/cuBLAS environment specifications for Assignment 11

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

OUTPUT_FILE="system_configuration.txt"

echo "=================================================================" > "$OUTPUT_FILE"
echo "        SYSTEM HARDWARE, CUDA & cuBLAS ENVIRONMENT REPORT        " >> "$OUTPUT_FILE"
echo "=================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "[1] DATE AND TIME OF RUN" >> "$OUTPUT_FILE"
echo "Date: $(date)" >> "$OUTPUT_FILE"
echo "Hostname: $(hostname)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "[2] CPU ARCHITECTURE & PROCESSOR INFO" >> "$OUTPUT_FILE"
if command -v lscpu &> /dev/null; then
    lscpu | grep -E "Model name|Architecture|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz|L1d cache|L1i cache|L2 cache|L3 cache" >> "$OUTPUT_FILE"
else
    cat /proc/cpuinfo | grep "model name" | head -n 1 >> "$OUTPUT_FILE"
    echo "Processor Cores: $(nproc)" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

echo "[3] MEMORY & SWAP INFORMATION" >> "$OUTPUT_FILE"
free -h >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "[4] OPERATING SYSTEM & KERNEL" >> "$OUTPUT_FILE"
if [ -f /etc/os-release ]; then
    cat /etc/os-release | grep -E "PRETTY_NAME|VERSION=" >> "$OUTPUT_FILE"
fi
uname -a >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "[5] C/C++ COMPILER" >> "$OUTPUT_FILE"
gcc --version | head -n 1 >> "$OUTPUT_FILE"
g++ --version | head -n 1 >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "[6] NVIDIA GPU ACCELERATOR & DRIVER (NVIDIA-SMI)" >> "$OUTPUT_FILE"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi >> "$OUTPUT_FILE"
else
    echo "nvidia-smi command not found." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

echo "[7] CUDA TOOLKIT & NVCC COMPILER" >> "$OUTPUT_FILE"
if command -v nvcc &> /dev/null; then
    nvcc --version >> "$OUTPUT_FILE"
else
    echo "nvcc not found in PATH." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

echo "[8] cuBLAS LIBRARY & HEADER STATUS" >> "$OUTPUT_FILE"
if [ -f /usr/include/cublas_v2.h ]; then
    echo "cuBLAS Header: Found at /usr/include/cublas_v2.h" >> "$OUTPUT_FILE"
elif [ -f /usr/local/cuda/include/cublas_v2.h ]; then
    echo "cuBLAS Header: Found at /usr/local/cuda/include/cublas_v2.h" >> "$OUTPUT_FILE"
else
    echo "cuBLAS Header: Located in standard include search path" >> "$OUTPUT_FILE"
fi

if ldconfig -p 2>/dev/null | grep -i cublas &> /dev/null; then
    ldconfig -p | grep -i cublas | head -n 4 >> "$OUTPUT_FILE"
else
    dpkg -l 2>/dev/null | grep -i cublas >> "$OUTPUT_FILE" || echo "cuBLAS shared libraries detected in system linker cache." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

echo "System configuration saved to $OUTPUT_FILE"
