#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1000

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0)
            printf("This program requires exactly 4 processes.\n");

        MPI_Finalize();
        return 1;
    }

    int cols_per_process = N / size;

    /*
       Every process needs the complete matrix A
       because each output column depends on all rows of A.
    */
    double *A = malloc((size_t)N * N * sizeof(double));

    double *local_B =
        malloc((size_t)N * cols_per_process * sizeof(double));

    double *local_C =
        malloc((size_t)N * cols_per_process * sizeof(double));

    if (A == NULL || local_B == NULL || local_C == NULL) {
        printf("Memory allocation failed on process %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /*
       Initialize A identically on every process.
    */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[(size_t)i * N + j] =
                (double)((i + j) % 100 + 1);
        }
    }

    /*
       Process 0 creates B in transposed form.

       B_transposed[j][k] = B[k][j]

       This makes columns of B contiguous so that
       MPI_Scatter can distribute complete columns.
    */
    double *B_transposed = NULL;
    double *C = NULL;

    if (rank == 0) {
        B_transposed =
            malloc((size_t)N * N * sizeof(double));

        C =
            malloc((size_t)N * N * sizeof(double));

        if (B_transposed == NULL || C == NULL) {
            printf("Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {

                double value =
                    (double)((i + j) % 100 + 1);

                B_transposed[(size_t)j * N + i] = value;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    /*
       Distribute columns of B.
    */
    MPI_Scatter(
        B_transposed,
        N * cols_per_process,
        MPI_DOUBLE,
        local_B,
        N * cols_per_process,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /*
       Each process computes its assigned columns of C.
    */
    for (int i = 0; i < N; i++) {

        for (int j = 0; j < cols_per_process; j++) {

            double sum = 0.0;

            for (int k = 0; k < N; k++) {

                sum +=
                    A[(size_t)i * N + k] *
                    local_B[(size_t)j * N + k];
            }

            local_C[(size_t)i * cols_per_process + j] = sum;
        }
    }

    /*
       Gather the column blocks back to process 0.
    */
    double *gathered_C = NULL;

    if (rank == 0) {
        gathered_C =
            malloc((size_t)N * N * sizeof(double));

        if (gathered_C == NULL) {
            printf("Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(
        local_C,
        N * cols_per_process,
        MPI_DOUBLE,
        gathered_C,
        N * cols_per_process,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double end = MPI_Wtime();

    /*
       Reconstruct C because each process supplied
       a consecutive block of columns.
    */
    if (rank == 0) {

        for (int p = 0; p < size; p++) {

            int start_col = p * cols_per_process;

            for (int i = 0; i < N; i++) {

                for (int j = 0; j < cols_per_process; j++) {

                    C[(size_t)i * N +
                      start_col + j] =
                        gathered_C[
                            (size_t)p * N * cols_per_process +
                            (size_t)i * cols_per_process +
                            j
                        ];
                }
            }
        }

        printf("Column-wise decomposition\n");
        printf("Matrix Size = %d x %d\n", N, N);
        printf("Processes = %d\n", size);
        printf("MPI Execution Time = %.6f seconds\n",
               end - start);

        printf("Sample Result: C[0][0] = %.2f, "
               "C[%d][%d] = %.2f\n",
               C[0],
               N - 1,
               N - 1,
               C[(size_t)(N - 1) * N + (N - 1)]);
    }

    free(A);
    free(local_B);
    free(local_C);

    if (rank == 0) {
        free(B_transposed);
        free(C);
        free(gathered_C);
    }

    MPI_Finalize();

    return 0;
}
