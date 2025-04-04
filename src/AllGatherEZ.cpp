#include <mpi.h>
#include <iostream>
#include <cstring>

using namespace std;

#define MiB1 1048576
#define WARM_UP 100
#define BENCHMARK_ITERATIONS 1000

static inline int copy_buffer_different_dt (const void *input_buffer, size_t scount,
    const MPI_Datatype sdtype, void *output_buffer,
    size_t rcount, const MPI_Datatype rdtype) {
if (input_buffer == NULL || output_buffer == NULL || scount <= 0 || rcount <= 0) {
return MPI_ERR_UNKNOWN;
}

int sdtype_size;
MPI_Type_size(sdtype, &sdtype_size);
int rdtype_size;
MPI_Type_size(rdtype, &rdtype_size);

size_t s_size = (size_t) sdtype_size * scount;
size_t r_size = (size_t) rdtype_size * rcount;

if (r_size < s_size) {
memcpy(output_buffer, input_buffer, r_size); // Copy as much as possible
return MPI_ERR_TRUNCATE;      // Indicate truncation
}

memcpy(output_buffer, input_buffer, s_size);        // Perform the memory copy

return MPI_SUCCESS;
}

double allgather_ring(const void *sbuf, size_t scount, MPI_Datatype sdtype,
void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
double start_time, end_time;
start_time = MPI_Wtime();
int line = -1, rank, size, sendto, recvfrom, i, recvdatafrom, senddatafrom;
ptrdiff_t rlb, rext;
char *tmpsend = NULL, *tmprecv = NULL;

MPI_Comm_size(comm, &size);
MPI_Comm_rank(comm, &rank);

MPI_Type_get_extent(rdtype, &rlb, &rext);

tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
if (MPI_IN_PLACE != sbuf) {
tmpsend = (char*) sbuf;
copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
}

sendto = (rank + 1) % size;
recvfrom  = (rank - 1 + size) % size;

for (i = 0; i < size - 1; i++) {

recvdatafrom = (rank - i - 1 + size) % size;
senddatafrom = (rank - i + size) % size;

tmprecv = (char*)rbuf + (ptrdiff_t)recvdatafrom * (ptrdiff_t)rcount * rext;
tmpsend = (char*)rbuf + (ptrdiff_t)senddatafrom * (ptrdiff_t)rcount * rext;

if(i == 0)
start_time = MPI_Wtime(); 

MPI_Sendrecv(tmpsend, rcount, rdtype, sendto, 0,
tmprecv, rcount, rdtype, recvfrom, 0,
comm, MPI_STATUS_IGNORE);
if(i == 0)
end_time = MPI_Wtime();
}
double total_time = end_time - start_time;
return total_time;

}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    double total_time = 0.0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int mib_count = 0;
    try {
        mib_count = stoi(argv[1]);  
    } catch (const invalid_argument& e) {
        cout << "Not valid argument!" << endl;
        return EXIT_FAILURE;
    }

    int BUFFER_SIZE = (mib_count * MiB1);
    int MSG_BUFFER_SIZE = BUFFER_SIZE / size;
    int msg_count = MSG_BUFFER_SIZE / sizeof(float);
    float *send_buffer = (float*) malloc(MSG_BUFFER_SIZE); 
    float *recv_buffer = (float*) malloc(BUFFER_SIZE);

    for (int i = 0; i < msg_count; i++) {
        send_buffer[i] = (float) rank; 
    }

    double start_time = MPI_Wtime();
    MPI_Allgather(send_buffer, msg_count, MPI_FLOAT, recv_buffer, msg_count, MPI_FLOAT, MPI_COMM_WORLD);
        //time_debug = allgather_ring(send_buffer, msg_count, MPI_FLOAT, recv_buffer, msg_count, MPI_FLOAT, MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    total_time += end_time - start_time;

    if(rank == 0){
        float buffer_gib = (BUFFER_SIZE / (float) (1024*1024*1024)) * 8;
        float bandwidth = (buffer_gib/size) * (size-1);
        bandwidth = bandwidth / total_time;
        cout << "Buffer: " << BUFFER_SIZE << " byte - " << buffer_gib << " Gib - " << mib_count << " MiB, Latency: " << total_time << ", Bandwidth: " << bandwidth << endl;
    }

    free(send_buffer);
    free(recv_buffer);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

