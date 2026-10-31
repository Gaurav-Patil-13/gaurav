#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int rank;
    int iterations = 1000;
    int message_size = 1024;

    char *buffer = malloc(message_size);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    for (int i = 0; i < iterations; i++)
    {
        if (rank == 0)
        {
            MPI_Send(buffer, message_size, MPI_CHAR, 1, 0,
                     MPI_COMM_WORLD);

            MPI_Recv(buffer, message_size, MPI_CHAR, 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (rank == 1)
        {
            MPI_Recv(buffer, message_size, MPI_CHAR, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Send(buffer, message_size, MPI_CHAR, 0, 0,
                     MPI_COMM_WORLD);
        }
    }

    double end = MPI_Wtime();

    if (rank == 0)
    {
        double total_time = end - start;
        double rtt = total_time / iterations;

        printf("Message Size: %d bytes\n", message_size);
        printf("Iterations: %d\n", iterations);
        printf("Total Time: %.9f seconds\n", total_time);
        printf("Average RTT: %.9f seconds\n", rtt);
    }

    free(buffer);

    MPI_Finalize();

    return 0;
}
