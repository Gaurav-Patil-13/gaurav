#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 10

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0)
            printf("This program requires exactly 3 processes.\n");

        MPI_Finalize();
        return 1;
    }

    /*
       Non-uniform row distribution:

       P0 -> 4 rows
       P1 -> 3 rows
       P2 -> 3 rows
    */
    int sendcounts[3] = {4, 3, 3};
    int displs[3] = {0, 4, 7};

    int local_rows = sendcounts[rank];

    double *A = NULL;
    double *X = malloc(N * sizeof(double));
    double *Y = NULL;

    double *local_A =
        malloc((size_t)local_rows * N * sizeof(double));

    double *local_Y =
        malloc(local_rows * sizeof(double));

    if (!X || !local_A || !local_Y) {
        printf("Memory allocation failed on process %d.\n",
               rank);

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /*
       Process 0 initializes A and X.
    */
    if (rank == 0) {

        A = malloc((size_t)N * N * sizeof(double));
        Y = malloc(N * sizeof(double));

        if (!A || !Y) {
            printf("Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /*
           Simple deterministic matrix.
        */
        for (int i = 0; i < N; i++) {
            X[i] = 1.0;

            for (int j = 0; j < N; j++) {
                A[(size_t)i * N + j] = j + 1;
            }
        }

        printf("Matrix Size = %d x %d\n", N, N);

        printf("Row distribution:\n");
        printf("Process 0 -> 4 rows\n");
        printf("Process 1 -> 3 rows\n");
        printf("Process 2 -> 3 rows\n");
    }

    /*
       Broadcast vector X to every process.
    */
    MPI_Bcast(
        X,
        N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /*
       Scatter variable numbers of rows.
    */
    MPI_Scatterv(
        A,
        (int[]){
            sendcounts[0] * N,
            sendcounts[1] * N,
            sendcounts[2] * N
        },
        (int[]){
            displs[0] * N,
            displs[1] * N,
            displs[2] * N
        },
        MPI_DOUBLE,
        local_A,
        local_rows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /*
       Local matrix-vector multiplication.
    */
    for (int i = 0; i < local_rows; i++) {

        local_Y[i] = 0.0;

        for (int j = 0; j < N; j++) {

            local_Y[i] +=
                local_A[(size_t)i * N + j] *
                X[j];
        }
    }

    /*
       Gather variable numbers of result elements.
    */
    MPI_Gatherv(
        local_Y,
        local_rows,
        MPI_DOUBLE,
        Y,
        sendcounts,
        displs,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    printf(
        "Process %d computed %d rows.\n",
        rank,
        local_rows
    );

    if (rank == 0) {

        printf("\nResult vector Y:\n");

        for (int i = 0; i < N; i++) {
            printf("%.2f ", Y[i]);
        }

        printf("\n");

        printf(
            "Verification: Y[0] = %.2f, Y[%d] = %.2f\n",
            Y[0],
            N - 1,
            Y[N - 1]
        );
    }

    free(X);
    free(local_A);
    free(local_Y);

    if (rank == 0) {
        free(A);
        free(Y);
    }

    MPI_Finalize();

    return 0;
}
