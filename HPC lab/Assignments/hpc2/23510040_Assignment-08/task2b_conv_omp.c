/*
 * Assignment 8 - Task 2B: OpenMP 2D Convolution
 * File: task2b_conv_omp.c
 *
 * Description: Parallelizes 2D Convolution using OpenMP multithreading.
 *              Evaluates static, dynamic, and guided loop scheduling strategies
 *              across 2, 4, 8 threads for grid sizes N = 1000, 2000, 4000.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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

void conv2d_openmp_static(const double *I, const double *K, double *O, int N, int threads) {
    int pad = KERNEL_SIZE / 2;
    #pragma omp parallel for num_threads(threads) schedule(static)
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

void conv2d_openmp_dynamic(const double *I, const double *K, double *O, int N, int threads) {
    int pad = KERNEL_SIZE / 2;
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 16)
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

void conv2d_openmp_guided(const double *I, const double *K, double *O, int N, int threads) {
    int pad = KERNEL_SIZE / 2;
    #pragma omp parallel for num_threads(threads) schedule(guided)
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

void run_conv_omp_benchmark(int N) {
    double *I = (double *)malloc((long)N * N * sizeof(double));
    double *O_seq = (double *)malloc((long)N * N * sizeof(double));
    double *O_omp = (double *)malloc((long)N * N * sizeof(double));
    double K[9] = {1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9};

    for (long i = 0; i < (long)N * N; i++) I[i] = (double)(i % 256);

    double seq_start = omp_get_wtime();
    conv2d_sequential(I, K, O_seq, N);
    double seq_time = omp_get_wtime() - seq_start;

    int threads_list[] = {2, 4, 8};
    for (int t = 0; t < 3; t++) {
        int threads = threads_list[t];

        // Static Schedule
        double t_stat_start = omp_get_wtime();
        conv2d_openmp_static(I, K, O_omp, N, threads);
        double t_stat = omp_get_wtime() - t_stat_start;
        double s_stat = seq_time / t_stat;

        // Dynamic Schedule
        double t_dyn_start = omp_get_wtime();
        conv2d_openmp_dynamic(I, K, O_omp, N, threads);
        double t_dyn = omp_get_wtime() - t_dyn_start;

        // Guided Schedule
        double t_guid_start = omp_get_wtime();
        conv2d_openmp_guided(I, K, O_omp, N, threads);
        double t_guid = omp_get_wtime() - t_guid_start;

        printf("%-10d %-8d %-15.6f %-12.6f %-12.6f %-12.6f %-10.4fx %-10.4f\n",
               N, threads, seq_time, t_stat, t_dyn, t_guid, s_stat, s_stat / threads);
    }

    free(I);
    free(O_seq);
    free(O_omp);
}

int main(void) {
    printf("====================================================================================================\n");
    printf(" Task 2B: OpenMP 2D Convolution Scheduling Strategy Benchmark\n");
    printf("====================================================================================================\n");
    printf("%-10s %-8s %-15s %-12s %-12s %-12s %-10s %-10s\n",
           "Grid Size", "Threads", "Sequential(s)", "Static(s)", "Dynamic(s)", "Guided(s)", "Speedup", "Efficiency");
    printf("----------------------------------------------------------------------------------------------------\n");

    run_conv_omp_benchmark(1000);
    run_conv_omp_benchmark(2000);
    run_conv_omp_benchmark(4000);

    printf("====================================================================================================\n\n");
    return 0;
}
