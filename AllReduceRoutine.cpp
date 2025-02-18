#include <mpi.h>
#include <iostream>
#include <cstring>

using namespace std;

#define MiB1 1048576
#define WARM_UP 10
#define BENCHMARK_ITERATIONS 100

#define COLL_BASE_COMPUTE_BLOCKCOUNT( COUNT, NUM_BLOCKS, SPLIT_INDEX,       \
                                       EARLY_BLOCK_COUNT, LATE_BLOCK_COUNT ) \
    EARLY_BLOCK_COUNT = LATE_BLOCK_COUNT = COUNT / NUM_BLOCKS;               \
    SPLIT_INDEX = COUNT % NUM_BLOCKS;                                        \
    if (0 != SPLIT_INDEX) {                                                  \
        EARLY_BLOCK_COUNT = EARLY_BLOCK_COUNT + 1;                           \
    }


static inline int copy_buffer(const void *input_buffer, void *output_buffer,
                              size_t count, const MPI_Datatype datatype) {
  if (input_buffer == NULL || output_buffer == NULL || count <= 0) {
    return MPI_ERR_UNKNOWN;
  }

  int datatype_size;
  MPI_Type_size(datatype, &datatype_size);                // Get the size of the MPI datatype

  size_t total_size = count * (size_t)datatype_size;

  memcpy(output_buffer, input_buffer, total_size);        // Perform the memory copy

  return MPI_SUCCESS;
}

int allreduce_ring(const void *sbuf, void *rbuf, size_t count, MPI_Datatype dtype,
                   MPI_Op op, MPI_Comm comm)
{
  int ret, line, rank, size, k, recv_from, send_to, block_count, inbi;
  int early_segcount, late_segcount, split_rank, max_segcount;
  char *tmpsend = NULL, *tmprecv = NULL, *inbuf[2] = {NULL, NULL};
  ptrdiff_t true_lb, true_extent, lb, extent;
  ptrdiff_t block_offset, max_real_segsize;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  ret = MPI_Comm_size(comm, &size);
  ret = MPI_Comm_rank(comm, &rank);


  /* Special case for size == 1 */
  if (1 == size) {
    if (MPI_IN_PLACE != sbuf) {
      ret = copy_buffer((char *) sbuf, (char *) rbuf, count, dtype);
    }
    return MPI_SUCCESS;
  }


  /* Allocate and initialize temporary buffers */
  ret = MPI_Type_get_extent(dtype, &lb, &extent);
  ret = MPI_Type_get_true_extent(dtype, &true_lb, &true_extent);

  /* Determine the number of elements per block and corresponding
     block sizes.
     The blocks are divided into "early" and "late" ones:
     blocks 0 .. (split_rank - 1) are "early" and
     blocks (split_rank) .. (size - 1) are "late".
     Early blocks are at most 1 element larger than the late ones.
  */
  COLL_BASE_COMPUTE_BLOCKCOUNT( count, size, split_rank,
                   early_segcount, late_segcount );
  max_segcount = early_segcount;
  max_real_segsize = true_extent + (max_segcount - 1) * extent;


  inbuf[0] = (char*)malloc(max_real_segsize);
  if (size > 2) {
    inbuf[1] = (char*)malloc(max_real_segsize);
  }

  /* Handle MPI_IN_PLACE */
  if (MPI_IN_PLACE != sbuf) {
    ret = copy_buffer((char *)sbuf, (char *) rbuf, count, dtype);
  }

  /* Computation loop */

  /*
     For each of the remote nodes:
     - post irecv for block (r-1)
     - send block (r)
     - in loop for every step k = 2 .. n
     - post irecv for block (r + n - k) % n
     - wait on block (r + n - k + 1) % n to arrive
     - compute on block (r + n - k + 1) % n
     - send block (r + n - k + 1) % n
     - wait on block (r + 1)
     - compute on block (r + 1)
     - send block (r + 1) to rank (r + 1)
     Note that we must be careful when computing the beginning of buffers and
     for send operations and computation we must compute the exact block size.
  */
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;

  inbi = 0;
  /* Initialize first receive from the neighbor on the left */
  ret = MPI_Irecv(inbuf[inbi], max_segcount, dtype, recv_from, 0, comm, &reqs[inbi]);
 
  /* Send first block (my block) to the neighbor on the right */
  block_offset = ((rank < split_rank)?
          ((ptrdiff_t)rank * (ptrdiff_t)early_segcount) :
          ((ptrdiff_t)rank * (ptrdiff_t)late_segcount + split_rank));
  block_count = ((rank < split_rank)? early_segcount : late_segcount);
  tmpsend = ((char*)rbuf) + block_offset * extent;
  ret = MPI_Send(tmpsend, block_count, dtype, send_to, 0, comm);


  for (k = 2; k < size; k++) {
    const int prevblock = (rank + size - k + 1) % size;

    inbi = inbi ^ 0x1;

    /* Post irecv for the current block */
    ret = MPI_Irecv(inbuf[inbi], max_segcount, dtype, recv_from, 0, comm, &reqs[inbi]);


    /* Wait on previous block to arrive */
    ret = MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);


    /* Apply operation on previous block: result goes to rbuf
       rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
    */
    block_offset = ((prevblock < split_rank)?
            ((ptrdiff_t)prevblock * early_segcount) :
            ((ptrdiff_t)prevblock * late_segcount + split_rank));
    block_count = ((prevblock < split_rank)? early_segcount : late_segcount);
    tmprecv = ((char*)rbuf) + (ptrdiff_t)block_offset * extent;
    MPI_Reduce_local(inbuf[inbi ^ 0x1], tmprecv, block_count, dtype, op);

    /* send previous block to send_to */
    ret = MPI_Send(tmprecv, block_count, dtype, send_to, 0, comm);

  }

  /* Wait on the last block to arrive */
  ret = MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);


  /* Apply operation on the last block (from neighbor (rank + 1)
     rbuf[rank+1] = inbuf[inbi] (op) rbuf[rank + 1] */
  recv_from = (rank + 1) % size;
  block_offset = ((recv_from < split_rank)?
          ((ptrdiff_t)recv_from * early_segcount) :
          ((ptrdiff_t)recv_from * late_segcount + split_rank));
  block_count = ((recv_from < split_rank)? early_segcount : late_segcount);
  tmprecv = ((char*)rbuf) + (ptrdiff_t)block_offset * extent;
  MPI_Reduce_local(inbuf[inbi], tmprecv, block_count, dtype, op);

  /* Distribution loop - variation of ring allgather */
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  for (k = 0; k < size - 1; k++) {
    const int recv_data_from = (rank + size - k) % size;
    const int send_data_from = (rank + 1 + size - k) % size;
    const int send_block_offset =
      ((send_data_from < split_rank)?
       ((ptrdiff_t)send_data_from * early_segcount) :
       ((ptrdiff_t)send_data_from * late_segcount + split_rank));
    const int recv_block_offset =
      ((recv_data_from < split_rank)?
       ((ptrdiff_t)recv_data_from * early_segcount) :
       ((ptrdiff_t)recv_data_from * late_segcount + split_rank));
    block_count = ((send_data_from < split_rank)?
             early_segcount : late_segcount);

    tmprecv = (char*)rbuf + (ptrdiff_t)recv_block_offset * extent;
    tmpsend = (char*)rbuf + (ptrdiff_t)send_block_offset * extent;

    ret = MPI_Sendrecv(tmpsend, block_count, dtype, send_to, 0,
                       tmprecv, max_segcount, dtype, recv_from,
                       0, comm, MPI_STATUS_IGNORE);

  }

  if (NULL != inbuf[0]) free(inbuf[0]);
  if (NULL != inbuf[1]) free(inbuf[1]);

  return MPI_SUCCESS;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(processor_name, &name_len);

    if (argc < 2) {
        cerr << "Please, insert an integer as argument" << endl;
        return 1;  
    }

    int mib_count = 0;
    try {
        mib_count = stoi(argv[1]);  
        cout << "Message is " << mib_count << " MiB" << endl;
    } catch (const invalid_argument& e) {
        cout << "Not valid argument!" << endl;
        return EXIT_FAILURE;
    }

    int msg_count = (mib_count * MiB1)/sizeof(float);
    int BUFFER_SIZE = (mib_count * MiB1);
    float *send_buffer = (float*) malloc(BUFFER_SIZE); 
    float *recv_buffer = (float*) malloc(BUFFER_SIZE);
    if (send_buffer == NULL || recv_buffer == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    for (int i = 0; i < msg_count; i++) {
        send_buffer[i] = (float) rank; 
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double total_time = 0.0;
    for(int i = 0; i < BENCHMARK_ITERATIONS + WARM_UP; ++i){

        double start_time = MPI_Wtime();
        //MPI_Allreduce(send_buffer, recv_buffer, msg_count, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        allreduce_ring(send_buffer, recv_buffer, msg_count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        double end_time = MPI_Wtime();

        if(i>WARM_UP) {
            total_time += end_time - start_time;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    total_time = (double)(total_time)/BENCHMARK_ITERATIONS;

    double min_time, max_time, avg_time;
    //MPI_Reduce(&total_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    //MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //MPI_Reduce(&total_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_time = avg_time/size;

    float verifier = 0;
    for(int i = 0; i<msg_count; i++){
        verifier += recv_buffer[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //if(rank == 0){
    float buffer_gib = (BUFFER_SIZE / (float) (1024*1024*1024)) * 8;
    float bandwidth =  2 * buffer_gib * ((size-1)/(float)size);
    bandwidth = bandwidth / total_time;
    cout << rank << " : "<< processor_name <<" : ALL REDUCE -> Buffer: "  << BUFFER_SIZE << " byte - " << buffer_gib << " Gib - " << mib_count << " MiB, verifier: " << verifier << ", Latency: " << total_time << ", Bandwidth: " << bandwidth << endl;
    //}

    free(send_buffer);
    free(recv_buffer);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

