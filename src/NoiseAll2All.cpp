#include <mpi.h>
#include <vector>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int small_buf_size = 16 * 1024;  // bytes per peer
    const int iterations = 10000;  // number of rounds

    // Each process will send a chunk to every other process
    std::vector<char> send_buf(small_buf_size * world_size, world_rank);
    std::vector<char> recv_buf(small_buf_size * world_size);

    while (1) {
        // Simulate background noise with all-to-all communication
        MPI_Alltoall(send_buf.data(), small_buf_size, MPI_CHAR,
                     recv_buf.data(), small_buf_size, MPI_CHAR,
                     MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}