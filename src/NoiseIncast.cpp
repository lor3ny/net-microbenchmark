#include "common.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int BUFFER_SIZE = 16 * 1024 * 1024;  // 2 MiB buffers

    unsigned char *buffer = (unsigned char*) malloc_align(BUFFER_SIZE); 
    if (buffer == NULL) {
        std::cerr << "Memory allocation failed!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    srand(time(NULL)*rank); 
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i] = rand()*rank % size; 
    }

    std::vector<MPI_Request> requests;

    while (1) {
        requests.clear();

        if (rank == 0) {

            for (int sender = 1; sender < size; ++sender) {
                MPI_Request req;
                MPI_Irecv(buffer, BUFFER_SIZE, MPI_BYTE, sender, 0, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }

        } else {
            MPI_Request req;
            MPI_Isend(buffer, BUFFER_SIZE, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    MPI_Finalize();
    return 0;
}