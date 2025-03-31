#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    const int N = 4; // total number of processes
    int sendbuf[N], recvbuf[N];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != N) {
        if (rank == 0)
            fprintf(stderr, "This program is meant to be run with %d processes.\n", N);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize send buffer: each value is rank * 10 + destination index
    for (int i = 0; i < N; i++) {
        sendbuf[i] = rank * 10 + i;
    }

    // Perform All-to-All communication
    MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);

    // Print send and receive buffers for each process
    printf("Process %d:\n", rank);
    printf("  Sent:    ");
    for (int i = 0; i < N; i++)
        printf("%d ", sendbuf[i]);
    printf("\n  Received:");
    for (int i = 0; i < N; i++)
        printf(" %d", recvbuf[i]);
    printf("\n");

    MPI_Finalize();
    return 0;
}
