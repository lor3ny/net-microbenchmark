#include <mpi.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int small_buf_size = 16;  // small buffer (bytes)
    const int iterations = 10000;   // number of rounds
    std::vector<char> send_buf(small_buf_size, world_rank);
    std::vector<char> recv_buf(small_buf_size);

    std::vector<MPI_Request> requests;

    while (1) {
        requests.clear();

        for (int peer = 0; peer < world_size; ++peer) {
            if (peer == world_rank) continue;

            // Post non-blocking send and receive to each peer
            MPI_Request send_req, recv_req;

            MPI_Isend(send_buf.data(), small_buf_size, MPI_CHAR, peer, 0, MPI_COMM_WORLD, &send_req);
            MPI_Irecv(recv_buf.data(), small_buf_size, MPI_CHAR, peer, 0, MPI_COMM_WORLD, &recv_req);

            requests.push_back(send_req);
            requests.push_back(recv_req);
        }

        // Wait for all communication to complete
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    MPI_Finalize();
    return 0;
}