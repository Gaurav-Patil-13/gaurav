/*
 * Assignment 8 - Task 1A: Sequential Matrix-Matrix Multiplication
 * File: task1a_matmul_seq.c
 *
 * Description: Computes matrix multiplication C = A * B sequentially for matrix sizes 
 *              N = 500, 1000, and 2000. Measures execution time using clock_gettime / omp_get_wtime.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matmul_sequential(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void run_benchmark(int N) {
    double *A = (double *)malloc((long)N * N * sizeof(double));
    double *B = (double *)malloc((long)N * N * sizeof(double));
    double *C = (double *)malloc((long)N * N * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed for N=%d\n", N);
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0 : 1.0;
            B[i * N + j] = (i == j) ? 1.0 : 0.5;
        }
    }

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    matmul_sequential(A, B, C, N);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double exec_time = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    printf("%-15d %-20.6f %-15.1f\n", N, exec_time, C[0]);

    free(A);
    free(B);
    free(C);
}

int main(void) {
    printf("=================================================================\n");
    printf(" Task 1A: Sequential Matrix-Matrix Multiplication Benchmark\n");
    printf("=================================================================\n");
    printf("%-15s %-20s %-15s\n", "Matrix Size (N)", "Execution Time (s)", "Check Value C[0]");
    printf("-----------------------------------------------------------------\n");

    run_benchmark(500);
    run_benchmark(1000);
    run_benchmark(2000);

    printf("=================================================================\n\n");
    return 0;
}
