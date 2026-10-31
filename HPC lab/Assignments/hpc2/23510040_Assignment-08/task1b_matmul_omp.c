/*
 * Assignment 8 - Task 1B: OpenMP Matrix-Matrix Multiplication
 * File: task1b_matmul_omp.c
 *
 * Description: Parallelizes matrix multiplication C = A * B using OpenMP shared-memory
 *              multithreading (#pragma omp parallel for). Benchmarks across 2, 4, 8 threads
 *              and matrix sizes N = 500, 1000, and 2000.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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

void matmul_openmp(const double *A, const double *B, double *C, int N, int num_threads) {
    #pragma omp parallel for num_threads(num_threads) schedule(static)
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

void run_omp_benchmark(int N, const int *thread_counts, int num_tests) {
    double *A = (double *)malloc((long)N * N * sizeof(double));
    double *B = (double *)malloc((long)N * N * sizeof(double));
    double *C_seq = (double *)malloc((long)N * N * sizeof(double));
    double *C_omp = (double *)malloc((long)N * N * sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0 : 1.0;
            B[i * N + j] = (i == j) ? 1.0 : 0.5;
        }
    }

    double seq_start = omp_get_wtime();
    matmul_sequential(A, B, C_seq, N);
    double seq_time = omp_get_wtime() - seq_start;

    for (int t = 0; t < num_tests; t++) {
        int threads = thread_counts[t];
        double omp_start = omp_get_wtime();
        matmul_openmp(A, B, C_omp, N, threads);
        double omp_time = omp_get_wtime() - omp_start;

        double speedup = (omp_time > 0.0) ? (seq_time / omp_time) : 0.0;
        double efficiency = speedup / threads;

        printf("%-10d %-10d %-18.6f %-18.6f %-12.4f %-12.4f\n",
               N, threads, seq_time, omp_time, speedup, efficiency);
    }

    free(A);
    free(B);
    free(C_seq);
    free(C_omp);
}

int main(void) {
    int thread_counts[] = {2, 4, 8};
    int num_tests = 3;

    printf("========================================================================================\n");
    printf(" Task 1B: OpenMP Parallel Matrix-Matrix Multiplication Benchmark\n");
    printf("========================================================================================\n");
    printf("%-10s %-10s %-18s %-18s %-12s %-12s\n",
           "Matrix Size", "Threads", "Sequential Time(s)", "OpenMP Time(s)", "Speedup (S)", "Efficiency");
    printf("----------------------------------------------------------------------------------------\n");

    run_omp_benchmark(500, thread_counts, num_tests);
    run_omp_benchmark(1000, thread_counts, num_tests);
    run_omp_benchmark(2000, thread_counts, num_tests);

    printf("========================================================================================\n\n");
    return 0;
}
