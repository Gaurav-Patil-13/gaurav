#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 500;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1)
        N = atoi(argv[1]);

    if (N % size != 0) {
        if (rank == 0)
            printf("Matrix size %d must be evenly divisible by %d processes.\n",
                   N, size);

        MPI_Finalize();
        return 1;
    }

    int rows_per_process = N / size;

    double *A = NULL;
    double *X = (double *)malloc(N * sizeof(double));
    double *Y = NULL;

    double *local_A =
        (double *)malloc((size_t)rows_per_process * N * sizeof(double));

    double *local_Y =
        (double *)malloc(rows_per_process * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc((size_t)N * N * sizeof(double));
        Y = (double *)malloc(N * sizeof(double));

        if (A == NULL || Y == NULL) {
            printf("Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < N; i++) {
            X[i] = 1.0;

            for (int j = 0; j < N; j++) {
                A[(size_t)i * N + j] =
                    (double)((i + j) % 100 + 1);
            }
        }
    }

    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    MPI_Scatter(
        A,
        rows_per_process * N,
        MPI_DOUBLE,
        local_A,
        rows_per_process * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    for (int i = 0; i < rows_per_process; i++) {
        local_Y[i] = 0.0;

        for (int j = 0; j < N; j++) {
            local_Y[i] +=
                local_A[(size_t)i * N + j] * X[j];
        }
    }

    MPI_Gather(
        local_Y,
        rows_per_process,
        MPI_DOUBLE,
        Y,
        rows_per_process,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Matrix Size = %d x %d\n", N, N);
        printf("Processes = %d\n", size);
        printf("MPI Execution Time = %.6f seconds\n", end - start);
        printf("Sample Result: Y[0] = %.2f, Y[%d] = %.2f\n",
               Y[0], N - 1, Y[N - 1]);
    }

    free(local_A);
    free(local_Y);
    free(X);

    if (rank == 0) {
        free(A);
        free(Y);
    }

    MPI_Finalize();

    return 0;
}
