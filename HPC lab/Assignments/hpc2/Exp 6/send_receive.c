#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank;
    char message[] = "Hello from Process 0";
    char received[100];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        MPI_Send(message, 100, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1)
    {
        MPI_Recv(received, 100, MPI_CHAR, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process 1 received: %s\n", received);
    }

    MPI_Finalize();

    return 0;
}
