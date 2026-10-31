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

    int rows = N / size;

    double *A = NULL;
    double *B = malloc((size_t)N * N * sizeof(double));
    double *C = NULL;

    double *local_A =
        malloc((size_t)rows * N * sizeof(double));

    double *local_C =
        calloc((size_t)rows * N, sizeof(double));

    if (!B || !local_A || !local_C) {
        printf("Memory allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        A = malloc((size_t)N * N * sizeof(double));
        C = malloc((size_t)N * N * sizeof(double));

        if (!A || !C) {
            printf("Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[(size_t)i * N + j] =
                    (i + j) % 100 + 1;

                B[(size_t)i * N + j] =
                    (i + j) % 100 + 1;
            }
        }
    }

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    MPI_Scatter(
        A,
        rows * N,
        MPI_DOUBLE,
        local_A,
        rows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < N; k++) {
            double a = local_A[(size_t)i * N + k];

            for (int j = 0; j < N; j++) {
                local_C[(size_t)i * N + j] +=
                    a * B[(size_t)k * N + j];
            }
        }
    }

    MPI_Gather(
        local_C,
        rows * N,
        MPI_DOUBLE,
        C,
        rows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Matrix Size = %d x %d\n", N, N);
        printf("Processes = %d\n", size);
        printf("MPI Execution Time = %.6f seconds\n", end - start);
        printf("Sample Result: C[0][0] = %.2f, C[%d][%d] = %.2f\n",
               C[0], N - 1, N - 1,
               C[(size_t)(N - 1) * N + (N - 1)]);
    }

    free(B);
    free(local_A);
    free(local_C);

    if (rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();

    return 0;
}
