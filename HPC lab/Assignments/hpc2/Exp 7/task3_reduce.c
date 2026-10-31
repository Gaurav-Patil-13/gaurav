#include <stdio.h>
#include <limits.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int array[16];
    int local_array[16];
    int local_sum = 0;
    int local_max = INT_MIN;
    int local_min = INT_MAX;
    int global_sum, global_max, global_min;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0)
            printf("This program requires exactly 4 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        for (int i = 0; i < 16; i++)
            array[i] = i + 1;

        printf("Original array: ");
        for (int i = 0; i < 16; i++)
            printf("%d ", array[i]);
        printf("\n");
    }

    MPI_Scatter(array, 4, MPI_INT,
                local_array, 4, MPI_INT,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < 4; i++) {
        local_sum += local_array[i];
        if (local_array[i] > local_max) local_max = local_array[i];
        if (local_array[i] < local_min) local_min = local_array[i];
    }

    printf("Process %d -> Sum=%d Max=%d Min=%d\n",
           rank, local_sum, local_max, local_min);

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nGlobal Sum = %d\n", global_sum);
        printf("Global Maximum = %d\n", global_max);
        printf("Global Minimum = %d\n", global_min);
    }

    MPI_Finalize();
    return 0;
}
