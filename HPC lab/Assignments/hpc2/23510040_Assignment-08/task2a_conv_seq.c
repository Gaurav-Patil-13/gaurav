/*
 * Assignment 8 - Task 2A: Sequential 2D Convolution
 * File: task2a_conv_seq.c
 *
 * Description: Computes 2D Convolution on large 2D grids (1000x1000, 2000x2000, 4000x4000)
 *              using a 3x3 kernel. Measures execution time sequentially.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define KERNEL_SIZE 3

void conv2d_sequential(const double *I, const double *K, double *O, int N) {
    int pad = KERNEL_SIZE / 2;
    for (int i = pad; i < N - pad; i++) {
        for (int j = pad; j < N - pad; j++) {
            double sum = 0.0;
            for (int u = -pad; u <= pad; u++) {
                for (int v = -pad; v <= pad; v++) {
                    sum += I[(i + u) * N + (j + v)] * K[(u + pad) * KERNEL_SIZE + (v + pad)];
                }
            }
            O[i * N + j] = sum;
        }
    }
}

void run_conv_benchmark(int N) {
    double *I = (double *)malloc((long)N * N * sizeof(double));
    double *O = (double *)malloc((long)N * N * sizeof(double));
    double K[9] = {1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9}; // 3x3 Box Blur Filter

    if (!I || !O) {
        fprintf(stderr, "Memory allocation failed for N=%d\n", N);
        exit(1);
    }

    for (long i = 0; i < (long)N * N; i++) {
        I[i] = (double)(i % 256);
        O[i] = 0.0;
    }

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    conv2d_sequential(I, K, O, N);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double exec_time = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    printf("%-15d %-20.6f %-15.2f\n", N, exec_time, O[(N/2) * N + (N/2)]);

    free(I);
    free(O);
}

int main(void) {
    printf("=================================================================\n");
    printf(" Task 2A: Sequential 2D Convolution Benchmark (3x3 Kernel)\n");
    printf("=================================================================\n");
    printf("%-15s %-20s %-15s\n", "Grid Size (N)", "Execution Time (s)", "Center Check Val");
    printf("-----------------------------------------------------------------\n");

    run_conv_benchmark(1000);
    run_conv_benchmark(2000);
    run_conv_benchmark(4000);

    printf("=================================================================\n\n");
    return 0;
}
