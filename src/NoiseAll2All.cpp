#include "common.hpp"


void all2all_memcpy(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm){

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int datatype_size;
    MPI_Type_size(sendtype, &datatype_size);

    const char* sbuf = static_cast<const char*>(sendbuf);
    char* rbuf = static_cast<char*>(recvbuf);

    double mem_time = MPI_Wtime(); 
    // Copy local data directly (self-send)
    std::memcpy(rbuf + rank * datatype_size * recvcount,
                sbuf + rank * datatype_size * sendcount,
                sendcount * datatype_size);

}

void custom_alltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                     void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int datatype_size;
    MPI_Type_size(sendtype, &datatype_size);

    const char* sbuf = static_cast<const char*>(sendbuf);
    char* rbuf = static_cast<char*>(recvbuf);

    std::vector<MPI_Request> requests;
    for (int i = 0; i < size; ++i) {
        if (i == rank) continue;

        MPI_Request req_recv;
        MPI_Request req_send;

        MPI_Isend(sbuf + i * datatype_size * sendcount, sendcount, sendtype, i, 0, comm, &req_send);
        MPI_Irecv(rbuf + i * datatype_size * recvcount, recvcount, recvtype, i, 0, comm, &req_recv);
        
        requests.push_back(req_send);
        requests.push_back(req_recv);
    }

    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int BUFFER_SIZE = 16 * 1024 * 1024;  // bytes per peer 16MiB

    // Each process will send a chunk to every other process
    unsigned char *send_buffer = (unsigned char*) malloc_align(BUFFER_SIZE*size); 
    unsigned char *recv_buffer = (unsigned char*) malloc_align(BUFFER_SIZE*size);
    if (send_buffer == NULL || recv_buffer == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    while (1) {
        all2all_memcpy(send_buf.data(), BUFFER_SIZE, MPI_BYTE, recv_buf.data(), BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);
        custom_alltoall(send_buf.datta(), BUFFER_SIZE, MPI_BYTE, recv_buf.data(), BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}