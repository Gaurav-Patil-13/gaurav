#!/bin/bash
# sys_info.sh - Collect system hardware, software, and CUDA environment specifications for Assignment 10

OUTPUT_FILE="system_configuration.txt"

echo "=================================================================" > "$OUTPUT_FILE"
echo "           SYSTEM HARDWARE & CUDA ENVIRONMENT REPORT             " >> "$OUTPUT_FILE"
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

echo "System configuration saved to $OUTPUT_FILE"
