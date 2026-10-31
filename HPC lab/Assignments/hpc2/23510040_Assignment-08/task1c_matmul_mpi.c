/*
 * Assignment 8 - Task 1C: MPI Matrix-Matrix Multiplication
 * File: task1c_matmul_mpi.c
 *
 * Description: Implements matrix multiplication C = A * B using MPI distributed-memory
 *              programming with row-wise domain decomposition. Matrix A rows are scattered
 *              via MPI_Scatter(), Matrix B is broadcast via MPI_Bcast(), and local computed
 *              rows of C are gathered at Root 0 via MPI_Gather().
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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

void run_mpi_benchmark(int N, int rank, int size) {
    if (N % size != 0) {
        if (rank == 0) {
            printf("Matrix size N=%d not divisible by P=%d processes. Skipping...\n", N, size);
        }
        return;
    }

    int local_rows = N / size;
    double *A = NULL;
    double *B = NULL;
    double *C_seq = NULL;
    double *C_mpi = NULL;

    if (rank == 0) {
        A = (double *)malloc((long)N * N * sizeof(double));
        B = (double *)malloc((long)N * N * sizeof(double));
        C_seq = (double *)malloc((long)N * N * sizeof(double));
        C_mpi = (double *)malloc((long)N * N * sizeof(double));

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (i == j) ? 2.0 : 1.0;
                B[i * N + j] = (i == j) ? 1.0 : 0.5;
            }
        }
    } else {
        B = (double *)malloc((long)N * N * sizeof(double));
    }

    double *local_A = (double *)malloc((long)local_rows * N * sizeof(double));
    double *local_C = (double *)malloc((long)local_rows * N * sizeof(double));

    // Sequential timing
    double seq_time = 0.0;
    if (rank == 0) {
        double t_seq_start = MPI_Wtime();
        matmul_sequential(A, B, C_seq, N);
        seq_time = MPI_Wtime() - t_seq_start;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // MPI parallel timing
    double mpi_start = MPI_Wtime();

    MPI_Scatter(A, local_rows * N, MPI_DOUBLE, local_A, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }

    MPI_Gather(local_C, local_rows * N, MPI_DOUBLE, C_mpi, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double mpi_time = MPI_Wtime() - mpi_start;

    if (rank == 0) {
        double speedup = (mpi_time > 0.0) ? (seq_time / mpi_time) : 0.0;
        double efficiency = speedup / size;

        printf("%-10d %-10d %-18.6f %-18.6f %-12.4f %-12.4f\n",
               N, size, seq_time, mpi_time, speedup, efficiency);

        free(A);
        free(C_seq);
        free(C_mpi);
    }

    free(B);
    free(local_A);
    free(local_C);
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("\n========================================================================================\n");
        printf(" Task 1C: MPI Parallel Matrix-Matrix Multiplication Benchmark\n");
        printf("========================================================================================\n");
        printf("%-10s %-10s %-18s %-18s %-12s %-12s\n",
               "Matrix Size", "Processes", "Sequential Time(s)", "MPI Time(s)", "Speedup (S)", "Efficiency");
        printf("----------------------------------------------------------------------------------------\n");
    }

    run_mpi_benchmark(500, rank, size);
    run_mpi_benchmark(1000, rank, size);
    run_mpi_benchmark(2000, rank, size);

    if (rank == 0) {
        printf("========================================================================================\n\n");
    }

    MPI_Finalize();
    return 0;
}
