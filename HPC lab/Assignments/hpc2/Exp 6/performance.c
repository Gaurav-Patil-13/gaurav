#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int rank;
    int sizes[] = {1024, 10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024};
    int num_sizes = 5;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < num_sizes; i++)
    {
        int size = sizes[i];
        char *buffer = malloc(size);

        if (rank == 0)
        {
            double start = MPI_Wtime();

            MPI_Send(buffer, size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

            double end = MPI_Wtime();

            printf("Message Size: %d bytes, Communication Time: %.9f seconds\n",
                   size, end - start);
        }
        else if (rank == 1)
        {
            MPI_Recv(buffer, size, MPI_CHAR, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        free(buffer);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
