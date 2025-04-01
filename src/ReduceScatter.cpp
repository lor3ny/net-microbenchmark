#include <mpi.h>
#include <iostream>
#include <cstring>

using namespace std;

#define B1 1
#define KiB1 1024
#define MiB1 1048576
#define GiB1 1073741824
#define WARM_UP 10
#define BENCHMARK_ITERATIONS 100


void noop(void *in, void *inout, int *len, MPI_Datatype *datatype) {
    return;
}
  
  
static inline int copy_buffer(const void *input_buffer, void *output_buffer,
    size_t count, const MPI_Datatype datatype) {

    int datatype_size;
    MPI_Type_size(datatype, &datatype_size);                // Get the size of the MPI datatype

    size_t total_size = count * (size_t)datatype_size;

    memcpy(output_buffer, input_buffer, total_size);        // Perform the memory copy

    return MPI_SUCCESS;
}
  

static inline ptrdiff_t datatype_span(MPI_Datatype dtype, size_t count, ptrdiff_t *gap) {
    if(count == 0) {
        *gap = 0;
        return 0;                                 // No memory span required for zero repetitions
    }

    MPI_Aint lb, extent;
    MPI_Aint true_lb, true_extent;

    // Get extend and true extent (true extent does not include padding)
    MPI_Type_get_extent(dtype, &lb, &extent);
    MPI_Type_get_true_extent(dtype, &true_lb, &true_extent);

    *gap = true_lb;                             // Store the true lower bound

    return true_extent + extent * (count - 1);  // Calculate the total memory span
}
  
  
double reduce_scatter_ring( const void *sbuf, void *rbuf, const int rcounts[],
    MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
    int ret, line, rank, size, i, k, recv_from, send_to;
    int inbi;
    size_t total_count, max_block_count;
    ptrdiff_t *displs = NULL;
    char *tmpsend = NULL, *tmprecv = NULL, *accumbuf = NULL, *accumbuf_free = NULL;
    char *inbuf_free[2] = {NULL, NULL}, *inbuf[2] = {NULL, NULL};
    ptrdiff_t extent, lb, max_real_segsize, dsize, gap = 0;
    MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    double total_time = 0.0;

    ret = MPI_Comm_size(comm, &size);
    ret = MPI_Comm_rank(comm, &rank);

    /* Determine the maximum number of elements per node,
    corresponding block size, and displacements array.
    */
    displs = (ptrdiff_t*) malloc(size * sizeof(ptrdiff_t));
    displs[0] = 0;
    total_count = rcounts[0];
    max_block_count = rcounts[0];
    for(i = 1; i < size; i++) {
        displs[i] = total_count;
        total_count += rcounts[i];
        if(max_block_count < rcounts[i]) max_block_count = rcounts[i];
    }

    /* Special case for size == 1 */
    if(1 == size) {
        if(MPI_IN_PLACE != sbuf) {
            ret = copy_buffer((char*) sbuf, (char*) rbuf,total_count, dtype);
        }
        free(displs);
        return MPI_SUCCESS;
    }

    /* Allocate and initialize temporary buffers, we need:
    - a temporary buffer to perform reduction (size total_count) since
    rbuf can be of rcounts[rank] size.
    - up to two temporary buffers used for communication/computation overlap.
    */
    ret = MPI_Type_get_extent(dtype, &lb, &extent);

    max_real_segsize = datatype_span(dtype, max_block_count, &gap);
    dsize = datatype_span(dtype, total_count, &gap);

    accumbuf_free = (char*)malloc(dsize);
    accumbuf = accumbuf_free - gap;

    inbuf_free[0] = (char*)malloc(max_real_segsize);
    inbuf[0] = inbuf_free[0] - gap;

    if(size > 2) {
        inbuf_free[1] = (char*)malloc(max_real_segsize);
        inbuf[1] = inbuf_free[1] - gap;
    }

    /* Handle MPI_IN_PLACE for size > 1 */
    if(MPI_IN_PLACE == sbuf) {
        sbuf = rbuf;
    }

    ret = copy_buffer((char*) sbuf, accumbuf, total_count, dtype);


    /* Computation loop */

    /*
    For each of the remote nodes:
    - post irecv for block (r-2) from (r-1) with wrap around
    - send block (r-1) to (r+1)
    - in loop for every step k = 2 .. n
    - post irecv for block (r - 1 + n - k) % n
    - wait on block (r + n - k) % n to arrive
    - compute on block (r + n - k ) % n
    - send block (r + n - k) % n
    - wait on block (r)
    - compute on block (r)
    - copy block (r) to rbuf
    Note that we must be careful when computing the beginning of buffers and
    for send operations and computation we must compute the exact block size.
    */

    send_to = (rank + 1) % size;
    recv_from = (rank + size - 1) % size;
    
    inbi = 0;
    /* Initialize first receive from the neighbor on the left */
    ret = MPI_Irecv(inbuf[inbi], max_block_count, dtype, recv_from, 0, comm, &reqs[inbi]);
    tmpsend = accumbuf + displs[recv_from] * extent;
    ret = MPI_Send(tmpsend, rcounts[recv_from], dtype, send_to, 0, comm);

    for(k = 2; k < size; k++) {
        const int prevblock = (rank + size - k) % size;

        inbi = inbi ^ 0x1;

        /* Post irecv for the current block */
        ret = MPI_Irecv(inbuf[inbi], max_block_count, dtype, recv_from, 0, comm, &reqs[inbi]);

        /* Wait on previous block to arrive */
        ret = MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);

        /* Apply operation on previous block: result goes to rbuf
        rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
        */
        tmprecv = accumbuf + displs[prevblock] * extent;
        MPI_Reduce_local(inbuf[inbi ^ 0x1], tmprecv, rcounts[prevblock], dtype, op);

        /* send previous block to send_to */
        ret = MPI_Send(tmprecv, rcounts[prevblock], dtype, send_to, 0, comm);
    }

    /* Wait on the last block to arrive */
    ret = MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);

    /* Apply operation on the last block (my block)
    rbuf[rank] = inbuf[inbi] (op) rbuf[rank] */
    tmprecv = accumbuf + displs[rank] * extent;
    //MPI_Reduce_local(inbuf[inbi], tmprecv, rcounts[rank], dtype, op);

    /* Copy result from tmprecv to rbuf */

    ret = copy_buffer(tmprecv, (char *)rbuf, rcounts[rank], dtype);

    if(NULL != displs) free(displs);
    if(NULL != accumbuf_free) free(accumbuf_free);
    if(NULL != inbuf_free[0]) free(inbuf_free[0]);
    if(NULL != inbuf_free[1]) free(inbuf_free[1]);

    return total_time;
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size, name_len, ret;
    double total_time = 0.0;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(processor_name, &name_len);

    if (argc < 3) {
        cerr << "Please, insert an integer as argument" << endl;
        return 1;  
    }

    int size_count = 0;
    try {
      size_count = stoi(argv[1]);  
    } catch (const invalid_argument& e) {
      cerr << "Not valid argument!" << endl;
      return EXIT_FAILURE;
    }

    char* size_type;
    long long int multiplier_type = B1;
    try {
      size_type = argv[2];  
      if(strcmp(size_type,"B") == 0){
        multiplier_type = B1;
      } else if(strcmp(size_type,"KiB") == 0){
        multiplier_type = KiB1;
      } else if(strcmp(size_type,"MiB") == 0){
        multiplier_type = MiB1;
      } else if(strcmp(size_type,"GiB") == 0){
        multiplier_type = GiB1;
      } else {
        cerr << "Second argument is not valid!" << endl;
        return EXIT_FAILURE;  
      }
    } catch (const invalid_argument& e) {
        cerr << "Not valid argument!" << endl;
        return EXIT_FAILURE;
    }

    if(size_count == 512 && strcmp(size_type, "B") == 0){
        cout << " {" << rank << " : "<< processor_name << "}" << endl;
    }


    int BUFFER_SIZE = (size_count * multiplier_type);
    int msg_count = BUFFER_SIZE/sizeof(float);
    int DATA_COUNT = (BUFFER_SIZE / sizeof(int));
    float *send_buffer = (float*) malloc(BUFFER_SIZE); 
    float *recv_buffer = (float*) malloc(BUFFER_SIZE/size);
    int *recvcounts = (int*) malloc(size);
    if (send_buffer == NULL || recv_buffer == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    for (int i = 0; i < size; i++) {
        recvcounts[i] = DATA_COUNT / size; 
    }


    for (int i = 0; i < msg_count; i++) {
        send_buffer[i] = (float) rank; 
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < BENCHMARK_ITERATIONS + WARM_UP; ++i){

        double start_time = MPI_Wtime();
        double remove_time = reduce_scatter_ring(send_buffer, recv_buffer, recvcounts, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        //MPI_Reduce_scatter(send_buffer, recv_buffer, recvcounts, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        double end_time = MPI_Wtime();

        if(i>WARM_UP) {
          total_time += end_time - start_time - remove_time;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    total_time = (double)(total_time)/BENCHMARK_ITERATIONS;

    double max_time;
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    float verifier = 0;
    for(int i = 0; i<msg_count/size; i++){
      verifier += recv_buffer[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
        float buffer_gib = (BUFFER_SIZE / (float) (1024*1024*1024)) * 8;
        float bandwidth = (buffer_gib/size) * (size-1);
        bandwidth = bandwidth / max_time;
        cout << "REDUCE-SCATTER Buffer: "  << BUFFER_SIZE << " byte - " << buffer_gib << " Gib - " << size_count << size_type << ", verifier: " << verifier << ", Latency: " << max_time << ", Bandwidth: " << bandwidth << endl;
    }

    free(send_buffer);
    free(recv_buffer);
    free(recvcounts);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

