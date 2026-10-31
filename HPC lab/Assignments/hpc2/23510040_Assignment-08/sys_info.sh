#!/bin/bash
# system_info.sh - Collect system hardware and software environment details for Assignment 8 benchmark report

OUTPUT_FILE="system_configuration.txt"

echo "=================================================================" > "$OUTPUT_FILE"
echo "        SYSTEM HARDWARE & MPI / OPENMP ENVIRONMENT REPORT        " >> "$OUTPUT_FILE"
echo "=================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "[1] DATE AND TIME OF RUN" >> "$OUTPUT_FILE"
echo "Date: $(date)" >> "$OUTPUT_FILE"
echo "Hostname: $(hostname)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "[2] CPU ARCHITECTURE & PROCESSOR INFO" >> "$OUTPUT_FILE"
if command -v lscpu &> /dev/null; then
    lscpu >> "$OUTPUT_FILE"
else
    cat /proc/cpuinfo | grep "model name" | head -n 1 >> "$OUTPUT_FILE"
    echo "Processor Cores: $(nproc)" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

echo "[3] MEMORY & SWAP INFORMATION" >> "$OUTPUT_FILE"
free -h >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "[4] OPERATING SYSTEM & KERNEL VERSION" >> "$OUTPUT_FILE"
if [ -f /etc/os-release ]; then
    cat /etc/os-release | grep -E "PRETTY_NAME|VERSION" >> "$OUTPUT_FILE"
fi
uname -a >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "[5] C COMPILER & OPENMP SUPPORT" >> "$OUTPUT_FILE"
gcc --version | head -n 1 >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "[6] MPI IMPLEMENTATION & VERSION" >> "$OUTPUT_FILE"
if command -v mpicc &> /dev/null; then
    mpicc --version | head -n 1 >> "$OUTPUT_FILE"
    mpirun --version | head -n 1 >> "$OUTPUT_FILE"
else
    echo "MPI (mpicc / mpirun) not found in system PATH." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

echo "System info collection completed. Saved to $OUTPUT_FILE"
