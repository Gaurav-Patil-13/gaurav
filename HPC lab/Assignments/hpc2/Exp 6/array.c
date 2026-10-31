#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank;
    int array[5] = {10, 20, 30, 40, 50};
    int received[5];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        MPI_Send(array, 5, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1)
    {
        MPI_Recv(received, 5, MPI_INT, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int sum = 0;
        int max = received[0];

        for (int i = 0; i < 5; i++)
        {
            sum += received[i];

            if (received[i] > max)
                max = received[i];
        }

        double average = sum / 5.0;

        printf("Received array: ");
        for (int i = 0; i < 5; i++)
            printf("%d ", received[i]);

        printf("\nSum = %d\n", sum);
        printf("Average = %.2f\n", average);
        printf("Maximum = %d\n", max);
    }

    MPI_Finalize();

    return 0;
}
