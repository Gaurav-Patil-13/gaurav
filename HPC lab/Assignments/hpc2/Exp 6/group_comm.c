#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int world_rank;
    MPI_Group world_group, new_group;
    MPI_Comm new_comm;

    int ranks[] = {0, 2, 4, 6};

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    MPI_Group_incl(world_group, 4, ranks, &new_group);

    MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &new_comm);

    if (new_comm != MPI_COMM_NULL)
    {
        int new_rank;
        MPI_Comm_rank(new_comm, &new_rank);

        printf("Global rank %d -> New communicator rank %d\n",
               world_rank, new_rank);

        MPI_Comm_free(&new_comm);
    }

    MPI_Group_free(&new_group);
    MPI_Group_free(&world_group);

    MPI_Finalize();

    return 0;
}
