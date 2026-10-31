#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank;
    int value = 100;
    int received;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&received, 1, MPI_INT, 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process 0 completed send and receive.\n");
    }
    else if (rank == 1)
    {
        MPI_Recv(&received, 1, MPI_INT, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        printf("Process 1 completed receive and send.\n");
    }

    MPI_Finalize();

    return 0;
}
