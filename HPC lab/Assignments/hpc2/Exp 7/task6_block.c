#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define N 1000
#define GRID 2

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

    const int block = N / GRID;

    /*
       Create a 2 x 2 Cartesian process grid:

             P0  P1
             P2  P3
    */
    int dims[2] = {GRID, GRID};
    int periods[2] = {0, 0};

    MPI_Comm cart_comm;

    MPI_Cart_create(
        MPI_COMM_WORLD,
        2,
        dims,
        periods,
        0,
        &cart_comm
    );

    int coords[2];

    MPI_Cart_coords(
        cart_comm,
        rank,
        2,
        coords
    );

    int row = coords[0];
    int col = coords[1];

    /*
       Row communicator:
       processes having the same row.

       Column communicator:
       processes having the same column.
    */
    MPI_Comm row_comm;
    MPI_Comm col_comm;

    int remain_row[2] = {0, 1};
    int remain_col[2] = {1, 0};

    MPI_Cart_sub(
        cart_comm,
        remain_row,
        &row_comm
    );

    MPI_Cart_sub(
        cart_comm,
        remain_col,
        &col_comm
    );

    /*
       Global matrices exist only on process 0.
    */
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    if (rank == 0) {

        A = malloc((size_t)N * N * sizeof(double));
        B = malloc((size_t)N * N * sizeof(double));
        C = calloc((size_t)N * N, sizeof(double));

        if (A == NULL || B == NULL || C == NULL) {
            printf("Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < N; i++) {

            for (int j = 0; j < N; j++) {

                A[(size_t)i * N + j] =
                    (double)((i + j) % 100 + 1);

                B[(size_t)i * N + j] =
                    (double)((i + j) % 100 + 1);
            }
        }
    }

    /*
       Each process initially receives:

       A(row, col)

       B(row, col)
    */
    double *local_A =
        malloc((size_t)block * block * sizeof(double));

    double *local_B =
        malloc((size_t)block * block * sizeof(double));

    double *A_broadcast =
        malloc((size_t)block * block * sizeof(double));

    double *B_broadcast =
        malloc((size_t)block * block * sizeof(double));

    double *local_C =
        calloc((size_t)block * block, sizeof(double));

    if (!local_A ||
        !local_B ||
        !A_broadcast ||
        !B_broadcast ||
        !local_C) {

        printf("Memory allocation failed on process %d.\n",
               rank);

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    /*
       Process 0 distributes the initial A and B blocks.
    */
    if (rank == 0) {

        for (int p = 0; p < size; p++) {

            int pcoords[2];

            MPI_Cart_coords(
                cart_comm,
                p,
                2,
                pcoords
            );

            double *temp_A =
                malloc((size_t)block * block *
                       sizeof(double));

            double *temp_B =
                malloc((size_t)block * block *
                       sizeof(double));

            for (int i = 0; i < block; i++) {

                for (int j = 0; j < block; j++) {

                    temp_A[(size_t)i * block + j] =
                        A[
                            (size_t)(pcoords[0] * block + i)
                            * N
                            + (pcoords[1] * block + j)
                        ];

                    temp_B[(size_t)i * block + j] =
                        B[
                            (size_t)(pcoords[0] * block + i)
                            * N
                            + (pcoords[1] * block + j)
                        ];
                }
            }

            if (p == 0) {

                memcpy(
                    local_A,
                    temp_A,
                    (size_t)block * block *
                    sizeof(double)
                );

                memcpy(
                    local_B,
                    temp_B,
                    (size_t)block * block *
                    sizeof(double)
                );

            } else {

                MPI_Send(
                    temp_A,
                    block * block,
                    MPI_DOUBLE,
                    p,
                    100,
                    MPI_COMM_WORLD
                );

                MPI_Send(
                    temp_B,
                    block * block,
                    MPI_DOUBLE,
                    p,
                    200,
                    MPI_COMM_WORLD
                );
            }

            free(temp_A);
            free(temp_B);
        }

    } else {

        MPI_Recv(
            local_A,
            block * block,
            MPI_DOUBLE,
            0,
            100,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        MPI_Recv(
            local_B,
            block * block,
            MPI_DOUBLE,
            0,
            200,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    /*
       SUMMA-style block multiplication.

       C(i,j) =
           A(i,0)B(0,j)
         + A(i,1)B(1,j)
    */
    for (int k = 0; k < GRID; k++) {

        /*
           Process (row,k) broadcasts its A block
           across its process row.
        */
        if (col == k) {

            memcpy(
                A_broadcast,
                local_A,
                (size_t)block * block *
                sizeof(double)
            );
        }

        MPI_Bcast(
            A_broadcast,
            block * block,
            MPI_DOUBLE,
            k,
            row_comm
        );

        /*
           Process (k,col) broadcasts its B block
           down its process column.
        */
        if (row == k) {

            memcpy(
                B_broadcast,
                local_B,
                (size_t)block * block *
                sizeof(double)
            );
        }

        MPI_Bcast(
            B_broadcast,
            block * block,
            MPI_DOUBLE,
            k,
            col_comm
        );

        /*
           Local block multiplication.
        */
        for (int i = 0; i < block; i++) {

            for (int x = 0; x < block; x++) {

                double a =
                    A_broadcast[
                        (size_t)i * block + x
                    ];

                for (int j = 0; j < block; j++) {

                    local_C[
                        (size_t)i * block + j
                    ] +=
                        a *
                        B_broadcast[
                            (size_t)x * block + j
                        ];
                }
            }
        }
    }

    /*
       Gather all C blocks back to process 0.
    */
    if (rank == 0) {

        for (int p = 0; p < size; p++) {

            int pcoords[2];

            MPI_Cart_coords(
                cart_comm,
                p,
                2,
                pcoords
            );

            double *temp_C =
                malloc((size_t)block * block *
                       sizeof(double));

            if (p == 0) {

                memcpy(
                    temp_C,
                    local_C,
                    (size_t)block * block *
                    sizeof(double)
                );

            } else {

                MPI_Recv(
                    temp_C,
                    block * block,
                    MPI_DOUBLE,
                    p,
                    300,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );
            }

            for (int i = 0; i < block; i++) {

                for (int j = 0; j < block; j++) {

                    C[
                        (size_t)(pcoords[0] * block + i)
                        * N
                        + (pcoords[1] * block + j)
                    ] =
                        temp_C[
                            (size_t)i * block + j
                        ];
                }
            }

            free(temp_C);
        }

    } else {

        MPI_Send(
            local_C,
            block * block,
            MPI_DOUBLE,
            0,
            300,
            MPI_COMM_WORLD
        );
    }

    double end = MPI_Wtime();

    if (rank == 0) {

        printf("Block-wise decomposition\n");
        printf("Process grid = 2 x 2\n");
        printf("Matrix Size = %d x %d\n", N, N);
        printf("Processes = %d\n", size);

        printf(
            "MPI Execution Time = %.6f seconds\n",
            end - start
        );

        printf(
            "Sample Result: C[0][0] = %.2f, "
            "C[%d][%d] = %.2f\n",
            C[0],
            N - 1,
            N - 1,
            C[(size_t)(N - 1) * N + (N - 1)]
        );
    }

    free(local_A);
    free(local_B);
    free(A_broadcast);
    free(B_broadcast);
    free(local_C);

    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();

    return 0;
}
