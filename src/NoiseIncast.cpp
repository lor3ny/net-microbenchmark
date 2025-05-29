#include "common.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int small_buf_size = 16 * 1024;  // bytes
    const int iterations = 10000;   // optional: number of rounds

    std::vector<char> buffer(small_buf_size, world_rank);
    std::vector<MPI_Request> requests;

    while (1) {
        requests.clear();

        if (world_rank == 0) {

            for (int sender = 1; sender < world_size; ++sender) {
                MPI_Request req;
                MPI_Irecv(buffer.data(), small_buf_size, MPI_CHAR, sender, 0, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }

        } else {
            MPI_Request req;
            MPI_Isend(buffer.data(), small_buf_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        // if (world_rank == 0){
        //     for(j=0;j<w_size-1;j++){
        //         MPI_Recv(&recv_buf[j*msg_size],recv_buf_size,MPI_BYTE, MPI_ANY_SOURCE
        //                 ,MPI_ANY_TAG, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //     }
        // }else{
        //     MPI_Send(send_buf,msg_size,MPI_BYTE,master_rank,my_rank,MPI_COMM_WORLD);
        // }
    }

    MPI_Finalize();
    return 0;
}