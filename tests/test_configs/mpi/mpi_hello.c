#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>


int main(int argc, char** argv) {
    sleep(1);
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char filename[64];
    sprintf(filename, "mpi_hello.%d.log", world_rank);
    FILE *log = fopen(filename, "w");

    fprintf(log, "Hello world from rank %d out of %d processors\n",
            world_rank, world_size);
    fflush(log);

    // unlink(filename);
    fclose(log);

    // Finalize the MPI environment.
    MPI_Finalize();
}
