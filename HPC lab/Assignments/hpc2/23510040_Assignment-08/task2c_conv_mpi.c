/*
 * Assignment 8 - Task 2C: MPI 2D Convolution with Boundary/Halo Row Exchange
 * File: task2c_conv_mpi.c
 *
 * Description: Implements 2D Convolution using MPI row-wise domain decomposition.
 *              Each process receives N/P rows of the input image. To compute boundary
 *              pixels, neighboring processes exchange top and bottom halo/ghost rows
 *              using point-to-point MPI_Sendrecv(). Results are gathered at Root 0.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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

void run_conv_mpi_benchmark(int N, int rank, int size) {
    if (N % size != 0) {
        if (rank == 0) printf("Grid size N=%d not divisible by P=%d processes. Skipping...\n", N, size);
        return;
    }

    int local_rows = N / size;
    int pad = KERNEL_SIZE / 2;

    double *I_global = NULL;
    double *O_seq = NULL;
    double *O_mpi = NULL;
    double K[9] = {1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9};

    if (rank == 0) {
        I_global = (double *)malloc((long)N * N * sizeof(double));
        O_seq = (double *)malloc((long)N * N * sizeof(double));
        O_mpi = (double *)malloc((long)N * N * sizeof(double));

        for (long i = 0; i < (long)N * N; i++) I_global[i] = (double)(i % 256);
    }

    // Allocate local buffer including 2 extra halo rows (top and bottom)
    int total_local_rows = local_rows + 2 * pad;
    double *local_I = (double *)malloc((long)total_local_rows * N * sizeof(double));
    double *local_O = (double *)malloc((long)local_rows * N * sizeof(double));

    // Sequential timing
    double seq_time = 0.0;
    if (rank == 0) {
        double t_seq_start = MPI_Wtime();
        conv2d_sequential(I_global, K, O_seq, N);
        seq_time = MPI_Wtime() - t_seq_start;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // MPI parallel timing
    double mpi_start = MPI_Wtime();

    // 1. Scatter main image rows into local_I offset by pad rows
    MPI_Scatter(I_global, local_rows * N, MPI_DOUBLE,
                &local_I[pad * N], local_rows * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // 2. Halo Row Exchange between neighboring ranks using MPI_Sendrecv
    int top_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int bottom_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Exchange with Top Neighbor (Send first compute row, receive top halo row)
    MPI_Sendrecv(&local_I[pad * N], N, MPI_DOUBLE, top_neighbor, 0,
                 &local_I[0], N, MPI_DOUBLE, top_neighbor, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Exchange with Bottom Neighbor (Send last compute row, receive bottom halo row)
    MPI_Sendrecv(&local_I[local_rows * N], N, MPI_DOUBLE, bottom_neighbor, 1,
                 &local_I[(local_rows + pad) * N], N, MPI_DOUBLE, bottom_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 3. Local 2D Convolution Computation
    for (int i = 0; i < local_rows; i++) {
        int global_row = rank * local_rows + i;
        if (global_row < pad || global_row >= N - pad) continue;

        for (int j = pad; j < N - pad; j++) {
            double sum = 0.0;
            for (int u = -pad; u <= pad; u++) {
                for (int v = -pad; v <= pad; v++) {
                    // Access local_I indexed with offset (i + pad + u)
                    sum += local_I[(i + pad + u) * N + (j + v)] * K[(u + pad) * KERNEL_SIZE + (v + pad)];
                }
            }
            local_O[i * N + j] = sum;
        }
    }

    // 4. Gather computed local output rows back at Root 0
    MPI_Gather(local_O, local_rows * N, MPI_DOUBLE,
               O_mpi, local_rows * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    double mpi_time = MPI_Wtime() - mpi_start;

    if (rank == 0) {
        double speedup = (mpi_time > 0.0) ? (seq_time / mpi_time) : 0.0;
        double efficiency = speedup / size;

        printf("%-10d %-10d %-18.6f %-18.6f %-12.4f %-12.4f\n",
               N, size, seq_time, mpi_time, speedup, efficiency);

        free(I_global);
        free(O_seq);
        free(O_mpi);
    }

    free(local_I);
    free(local_O);
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("\n========================================================================================\n");
        printf(" Task 2C: MPI 2D Convolution with Boundary/Halo Row Exchange Benchmark\n");
        printf("========================================================================================\n");
        printf("%-10s %-10s %-18s %-18s %-12s %-12s\n",
               "Grid Size", "Processes", "Sequential Time(s)", "MPI Time(s)", "Speedup (S)", "Efficiency");
        printf("----------------------------------------------------------------------------------------\n");
    }

    run_conv_mpi_benchmark(1000, rank, size);
    run_conv_mpi_benchmark(2000, rank, size);
    run_conv_mpi_benchmark(4000, rank, size);

    if (rank == 0) {
        printf("========================================================================================\n\n");
    }

    MPI_Finalize();
    return 0;
}
