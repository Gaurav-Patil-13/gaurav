/*
 * Assignment 8 - Task 3: OpenMP vs MPI Comparative Performance Benchmark Driver
 * File: task3_comparative_benchmark.c
 *
 * Description: Directly compares Sequential, OpenMP shared-memory multithreading, and MPI
 *              distributed-memory message passing side-by-side for equivalent workloads.
 *              Evaluates Application 1 (Matrix Multiplication 1000x1000) and
 *              Application 2 (2D Convolution 2000x2000).
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#define MAT_N 1000
#define CONV_N 2000
#define KERNEL_SIZE 3

// Sequential Matrix Multiplication
void matmul_seq(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

// OpenMP Matrix Multiplication
void matmul_omp(const double *A, const double *B, double *C, int N, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

// Sequential 2D Convolution
void conv_seq(const double *I, const double *K, double *O, int N) {
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

// OpenMP 2D Convolution
void conv_omp(const double *I, const double *K, double *O, int N, int threads) {
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

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("\n========================================================================================\n");
        printf(" Task 3: OpenMP vs MPI Performance Side-by-Side Comparison\n");
        printf("========================================================================================\n");
        printf("%-20s %-12s %-12s %-16s %-12s %-12s\n",
               "Application", "Model", "Cores/Ranks", "Execution Time(s)", "Speedup (S)", "Efficiency");
        printf("----------------------------------------------------------------------------------------\n");
    }

    // Benchmark Application 1: Matrix Multiplication (1000x1000)
    double *A = NULL, *B = NULL, *C = NULL;
    if (rank == 0) {
        A = (double *)malloc((long)MAT_N * MAT_N * sizeof(double));
        B = (double *)malloc((long)MAT_N * MAT_N * sizeof(double));
        C = (double *)malloc((long)MAT_N * MAT_N * sizeof(double));
        for (long i = 0; i < (long)MAT_N * MAT_N; i++) { A[i] = 1.0; B[i] = 2.0; }
    }

    double t_mat_seq = 0.0;
    if (rank == 0) {
        double s = omp_get_wtime();
        matmul_seq(A, B, C, MAT_N);
        t_mat_seq = omp_get_wtime() - s;
        printf("%-20s %-12s %-12d %-16.6f %-12.4f %-12.4f\n",
               "1. MatMul (1000x1000)", "Sequential", 1, t_mat_seq, 1.0, 1.0);

        // OpenMP run with matching thread count
        double s_omp = omp_get_wtime();
        matmul_omp(A, B, C, MAT_N, size);
        double t_mat_omp = omp_get_wtime() - s_omp;
        printf("%-20s %-12s %-12d %-16.6f %-12.4f %-12.4f\n",
               "1. MatMul (1000x1000)", "OpenMP", size, t_mat_omp, t_mat_seq / t_mat_omp, (t_mat_seq / t_mat_omp) / size);
    }

    // MPI Matrix Multiplication
    int local_rows_mat = MAT_N / size;
    double *B_mpi = (double *)malloc((long)MAT_N * MAT_N * sizeof(double));
    double *local_A = (double *)malloc((long)local_rows_mat * MAT_N * sizeof(double));
    double *local_C = (double *)malloc((long)local_rows_mat * MAT_N * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    double s_mpi_mat = MPI_Wtime();
    MPI_Scatter(A, local_rows_mat * MAT_N, MPI_DOUBLE, local_A, local_rows_mat * MAT_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast((rank == 0) ? B : B_mpi, MAT_N * MAT_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows_mat; i++) {
        for (int j = 0; j < MAT_N; j++) {
            double sum = 0.0;
            for (int k = 0; k < MAT_N; k++) sum += local_A[i * MAT_N + k] * ((rank == 0) ? B[k * MAT_N + j] : B_mpi[k * MAT_N + j]);
            local_C[i * MAT_N + j] = sum;
        }
    }
    MPI_Gather(local_C, local_rows_mat * MAT_N, MPI_DOUBLE, C, local_rows_mat * MAT_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_mat_mpi = MPI_Wtime() - s_mpi_mat;

    if (rank == 0) {
        printf("%-20s %-12s %-12d %-16.6f %-12.4f %-12.4f\n",
               "1. MatMul (1000x1000)", "MPI", size, t_mat_mpi, t_mat_seq / t_mat_mpi, (t_mat_seq / t_mat_mpi) / size);
        printf("----------------------------------------------------------------------------------------\n");
    }

    free(B_mpi);
    free(local_A);
    free(local_C);
    if (rank == 0) { free(A); free(B); free(C); }

    // Benchmark Application 2: 2D Convolution (2000x2000)
    double *I = NULL, *O = NULL;
    double K[9] = {1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9};

    if (rank == 0) {
        I = (double *)malloc((long)CONV_N * CONV_N * sizeof(double));
        O = (double *)malloc((long)CONV_N * CONV_N * sizeof(double));
        for (long i = 0; i < (long)CONV_N * CONV_N; i++) I[i] = (double)(i % 256);
    }

    double t_conv_seq = 0.0;
    if (rank == 0) {
        double s = omp_get_wtime();
        conv_seq(I, K, O, CONV_N);
        t_conv_seq = omp_get_wtime() - s;
        printf("%-20s %-12s %-12d %-16.6f %-12.4f %-12.4f\n",
               "2. Conv2D (2000x2000)", "Sequential", 1, t_conv_seq, 1.0, 1.0);

        double s_omp = omp_get_wtime();
        conv_omp(I, K, O, CONV_N, size);
        double t_conv_omp = omp_get_wtime() - s_omp;
        printf("%-20s %-12s %-12d %-16.6f %-12.4f %-12.4f\n",
               "2. Conv2D (2000x2000)", "OpenMP", size, t_conv_omp, t_conv_seq / t_conv_omp, (t_conv_seq / t_conv_omp) / size);
    }

    // MPI 2D Convolution
    int local_rows_conv = CONV_N / size;
    int pad = KERNEL_SIZE / 2;
    double *local_I = (double *)malloc((long)(local_rows_conv + 2 * pad) * CONV_N * sizeof(double));
    double *local_O = (double *)malloc((long)local_rows_conv * CONV_N * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    double s_mpi_conv = MPI_Wtime();

    MPI_Scatter(I, local_rows_conv * CONV_N, MPI_DOUBLE, &local_I[pad * CONV_N], local_rows_conv * CONV_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int top_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int bottom_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    MPI_Sendrecv(&local_I[pad * CONV_N], CONV_N, MPI_DOUBLE, top_neighbor, 0, &local_I[0], CONV_N, MPI_DOUBLE, top_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&local_I[local_rows_conv * CONV_N], CONV_N, MPI_DOUBLE, bottom_neighbor, 1, &local_I[(local_rows_conv + pad) * CONV_N], CONV_N, MPI_DOUBLE, bottom_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < local_rows_conv; i++) {
        int global_row = rank * local_rows_conv + i;
        if (global_row < pad || global_row >= CONV_N - pad) continue;
        for (int j = pad; j < CONV_N - pad; j++) {
            double sum = 0.0;
            for (int u = -pad; u <= pad; u++) {
                for (int v = -pad; v <= pad; v++) {
                    sum += local_I[(i + pad + u) * CONV_N + (j + v)] * K[(u + pad) * KERNEL_SIZE + (v + pad)];
                }
            }
            local_O[i * CONV_N + j] = sum;
        }
    }

    MPI_Gather(local_O, local_rows_conv * CONV_N, MPI_DOUBLE, O, local_rows_conv * CONV_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_conv_mpi = MPI_Wtime() - s_mpi_conv;

    if (rank == 0) {
        printf("%-20s %-12s %-12d %-16.6f %-12.4f %-12.4f\n",
               "2. Conv2D (2000x2000)", "MPI", size, t_conv_mpi, t_conv_seq / t_conv_mpi, (t_conv_seq / t_conv_mpi) / size);
        printf("========================================================================================\n\n");
        free(I); free(O);
    }

    free(local_I);
    free(local_O);

    MPI_Finalize();
    return 0;
}
