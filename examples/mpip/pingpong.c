/* Minimal MPI ping-pong for mpiP documentation (not built by default). */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    int rank, size;
    double x = 1.0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size >= 2) {
        if (rank == 0) {
            MPI_Send(&x, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&x, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(&x, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&x, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}
