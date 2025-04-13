#include <mpi.h>
#include <iostream>
#include <cstring>
#include <climits>
#include <cassert>
#include <cuda_runtime.h>
#include <unordered_map>
#include <omp.h>

using namespace std;

#define B1 1
#define KiB1 1024
#define MiB1 1048576
#define GiB1 1073741824
#define WARM_UP 10

#define LIBSWING_MAX_SUPPORTED_DIMENSIONS 3 // We support up to 3D torus
#define LIBSWING_MAX_STEPS 20

static int rhos[LIBSWING_MAX_STEPS] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341, 683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};
/*
static int smallest_negabinary[LIBSWING_MAX_STEPS] = {0, 0, -2, -2, -10, -10, -42, -42,
  -170, -170, -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762};
static int largest_negabinary[LIBSWING_MAX_STEPS] = {0, 1, 1, 5, 5, 21, 21, 85, 85,
  341, 341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525};
*/


#define CUDA_CHECK(cmd) do {                        \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


inline int pi(int rank, int step, int comm_sz) {
  int dest;

  if((rank & 1) == 0) dest = (int)(rank + rhos[step]) % comm_sz;//(rank + rhos[step]) % comm_sz;  // Even rank
  else dest = (int)(rank - rhos[step]) % comm_sz; //(rank - rhos[step]) % comm_sz;                 // Odd rank

  if(dest < 0) dest += comm_sz;                              // Adjust for negative ranks

  return dest;
}

inline int hibit(int value, int start)
{
  unsigned int mask;
  /* Only look at the part that the caller wanted looking at */
  mask = value & ((1 << start) - 1);

  if(0 == mask) {
    return -1;
  }

  start = (8 * sizeof(int) - 1) - __builtin_clz(mask);

  return start;
}


__global__ void reduce_sum_kernel(const int *in, int *inout, size_t count) {

  size_t global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t thread_count = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
  size_t i, idx;
  for(i = 0; global_thread_idx + i*thread_count < count; i++){
    idx = global_thread_idx + i*thread_count; 
    inout[idx] += in[idx]; 
  }
}

__global__ void reduce_sum_kernel_step0(const int *inA, const int *inB, int *out, size_t count) { 
  size_t global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t thread_count = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
  size_t i, idx;

  for(i = 0; global_thread_idx + i*thread_count < count; i++){
    idx = global_thread_idx + i*thread_count; 
    out[idx] = inA[idx] + inB[idx]; 
  }
}

inline int log_2(int value) {
    if (1 > value) {
        return -1;
    }
    return sizeof(int)*8 - 1 - __builtin_clz(value);
}

int allreduce_swing_lat(const void *sbuf, void *rbuf, size_t count, MPI_Datatype dtype, MPI_Op op, MPI_Comm comm, char *tmp_buf) {
  
  int rank, size, datatype_size;
  char *tmpsend, *tmprecv;
  //char *inplacebuf_free; 
  
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dtype, &datatype_size);

  /*
  // Special case for size == 1
  if(1 == size) {
    if(MPI_IN_PLACE != sbuf) { 
      //memcpy(rbuf, sbuf, count * datatype_size);
      CUDA_CHECK(cudaMemcpy(rbuf, sbuf, count * datatype_size, cudaMemcpyDeviceToDevice));
    }
    return MPI_SUCCESS;
  }

  // Allocate and initialize temporary send buffer
  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_get_true_extent(dtype, &gap, &true_extent);
  span = true_extent + extent * (count - 1);
  //inplacebuf_free = (char*) malloc(span + gap);
  CUDA_CHECK(cudaMalloc((void**) &inplacebuf_free, span+gap));
  char *inplacebuf = inplacebuf_free + gap;

  // Copy content from sbuffer to inplacebuf
  if (MPI_IN_PLACE == sbuf) {
    CUDA_CHECK(cudaMemcpy(inplacebuf, rbuf, count * datatype_size, cudaMemcpyDeviceToDevice));
  } else {
    CUDA_CHECK(cudaMemcpy(inplacebuf, sbuf, count * datatype_size, cudaMemcpyDeviceToDevice));
  }
  */

  tmpsend = (char*) sbuf; //tmp_buf
  tmprecv = (char*) rbuf;
  int steps = log_2(size);
  
  // Determine nearest power of two less than or equal to size
  // and return an error if size is 0
  /*
  int steps = hibit(size, (int)(sizeof(size) * CHAR_BIT) - 1);
  if(steps == -1) {
      return MPI_ERR_ARG;
  }
  int adjsize = 1 << steps;  // Largest power of two <= size

  // Number of nodes that exceed the largest power of two less than or equal to size
  int extra_ranks = size - adjsize;
  int is_power_of_two = (size & (size - 1)) == 0;

  // First part of computation to get a 2^n number of nodes.
  // What happens is that first #extra_rank even nodes sends their
  // data to the successive node and do not partecipate in the general
  // collective call operation.
  // All the nodes that do not stop their computation will receive an alias
  // called new_node, used to calculate their correct destination wrt this
  // new "cut" topology.
  
  int new_rank = rank, loop_flag = 0;
  if(rank <  (2 * extra_ranks)) {
    if(0 == (rank % 2)) {
      ret = MPI_Send(tmpsend, count, dtype, (rank + 1), 0, comm);
      if(MPI_SUCCESS != ret) { 
        cerr << "ERROR: Redscat phase, send section." << endl;
        return -1;
      }
      loop_flag = 1;
    } else {
      ret = MPI_Recv(tmprecv, count, dtype, (rank - 1), 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != ret) { 
        cerr << "ERROR: Redscat phase, recv section." << endl;
        return -1;
      }


      reduce_sum_kernel<<<512, 512>>>((const int*)tmprecv, (int*)tmpsend, count);
      new_rank = rank >> 1;
    }
  } else new_rank = rank - extra_ranks;
  CUDA_CHECK(cudaDeviceSynchronize());
  */

  // Actual allreduce computation for general cases
  // If the extra_ranks phasee has been done, this phase won't be executed. 
  // I'm running with 4 nodes, so the first phase won't be done but this yes.
  int s, vdest, dest;
  for(s = 0; s < steps; s++){
    //if(loop_flag) break;
    vdest = pi(rank /*new_rank*/, s, size/*adjsize*/);

    /*
    dest = is_power_of_two ?
              vdest :
              (vdest < extra_ranks) ?
              (vdest << 1) + 1 : vdest + extra_ranks;
    */
    dest = vdest;
    
    if(s>0){
      tmpsend = (char*) rbuf;
    }

    MPI_Sendrecv(tmpsend, count, dtype, dest, 0,
                       tmp_buf, count, dtype, dest, 0,
                       comm, MPI_STATUS_IGNORE);

    size_t current_segment_size = count;
    if(s == 0){        
      //char* tmp_send_0 = (char *) send_buf + dest * datatype_size; //+ offset * datatype_size;
      reduce_sum_kernel_step0<<<512, 512>>>((const int*)tmp_buf, (int*)tmpsend, (int*)tmprecv, current_segment_size);
    } else {        
      reduce_sum_kernel<<<512, 512>>>((const int*)tmp_buf, (int*)tmprecv, current_segment_size);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Final results is sent to nodes that are not included in general computation
  // (general computation loop requires 2^n nodes).
  /*
  if(rank < (2 * extra_ranks)){
    if(!loop_flag){
      ret = MPI_Send(tmpsend, count, dtype, (rank - 1), 0, comm);
    } else {
      ret = MPI_Recv(rbuf, count, dtype, (rank + 1), 0, comm, MPI_STATUS_IGNORE);
      tmpsend = (char*)rbuf;
    }
  }
  */

  /*
  if(tmpsend != rbuf) {
    CUDA_CHECK(cudaMemcpy(rbuf, tmpsend, count * datatype_size, cudaMemcpyDeviceToDevice));
  }
  */

  //CUDA_CHECK(cudaFree(inplacebuf_free));
  return MPI_SUCCESS;
}


__global__ void sum4arrays(const int* a, const int* b, const int* c, const int* d, int* out, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < count; i += stride) {
      out[i] = a[i] + b[i] + c[i] + d[i];
  }
}

int intra_reducescatter_block(void *sendbuf, void *recvbuf, size_t recvcount, MPI_Datatype recvtype, MPI_Comm comm){
    // Do a reduce-scatter where each rank isends and irecvs from everyone else
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int datatype_size;
    MPI_Type_size(recvtype, &datatype_size);
    MPI_Request* send_req = (MPI_Request*) malloc(sizeof(MPI_Request) * size);
    MPI_Request* recv_req = (MPI_Request*) malloc(sizeof(MPI_Request) * size);
    int next_send_req = 0, next_recv_req = 0;
    for (int i = 0; i < size; i++) {
        if (i != rank) {
            MPI_Isend((char*) sendbuf + i * recvcount * datatype_size, recvcount, recvtype, i, 0, comm, &send_req[next_send_req]);
            ++next_send_req;
            MPI_Irecv((char*) recvbuf + i * recvcount * datatype_size, recvcount, recvtype, i, 0, comm, &recv_req[next_recv_req]);
            ++next_recv_req;
            //printf("Rank %d setting tris_ptr[%d] to offset %d\n", rank, next_tris_ptr, i*recvcount * datatype_size);
        }
    } 
    MPI_Waitall(next_recv_req, recv_req, MPI_STATUSES_IGNORE);
    //printf("Rank %d setting tris_ptr[%d] to offset %d recvcount: %d \n", rank, next_tris_ptr, rank*recvcount * datatype_size, recvcount);
    sum4arrays<<<512, 512>>>((const int*) ((rank == 0 ? (char*) sendbuf : (char*) recvbuf) + 0 * recvcount * datatype_size), 
                             (const int*) ((rank == 1 ? (char*) sendbuf : (char*) recvbuf) + 1 * recvcount * datatype_size),
                             (const int*) ((rank == 2 ? (char*) sendbuf : (char*) recvbuf) + 2 * recvcount * datatype_size),
                             (const int*) ((rank == 3 ? (char*) sendbuf : (char*) recvbuf) + 3 * recvcount * datatype_size),
                             (int*) ((char*) recvbuf + ((rank + 1) % 4) * recvcount * datatype_size), recvcount);
    MPI_Waitall(next_send_req, send_req, MPI_STATUSES_IGNORE);
    free(send_req);
    free(recv_req);        
    cudaDeviceSynchronize();
    return MPI_SUCCESS;
}

int intra_reducescatter_block_segmented(void *sendbuf, void *recvbuf, size_t recvcount, MPI_Datatype recvtype, MPI_Comm comm, size_t segment_size) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int datatype_size;
  MPI_Type_size(recvtype, &datatype_size);

  int segment_count = segment_size/datatype_size;
  int num_segments = ceil(recvcount / (double) segment_count);

  //int num_segments = 1;
  //int segment_size = (recvcount + num_segments - 1) / num_segments;  // round up
  int last_segment_count = recvcount - segment_count * (num_segments - 1);


  MPI_Request* send_req = (MPI_Request*) malloc(sizeof(MPI_Request) * size * num_segments);
  MPI_Request* recv_req = (MPI_Request*) malloc(sizeof(MPI_Request) * size * num_segments);
  int next_send_req = 0, next_recv_req = 0;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  for (int seg = 0; seg < num_segments; ++seg) {
      int curr_seg_count = (seg == num_segments - 1) ? last_segment_count : segment_count;
      //size_t seg_bytes = curr_seg_size * datatype_size;

      for (int peer = 0; peer < size; ++peer) {
          if (peer != rank) {
              // offsets in bytes
              size_t offset = (peer * recvcount + seg * segment_count) * datatype_size;

              MPI_Isend((char*)sendbuf + offset, curr_seg_count, recvtype, peer, seg, comm, &send_req[next_send_req++]);
              MPI_Irecv((char*)recvbuf + offset, curr_seg_count, recvtype, peer, seg, comm, &recv_req[next_recv_req++]);
          }
      }
  }

  // Wait for each segment to arrive from all peers, then reduce
  for (int seg = 0; seg < num_segments; ++seg) {
    int curr_seg_count = (seg == num_segments - 1) ? last_segment_count : segment_count;

      MPI_Waitall(size - 1, &recv_req[seg * (size - 1)], MPI_STATUSES_IGNORE);

      // Pointers for this segment
      const int* a = (const int*)((rank == 0 ? (char*) sendbuf : (char*) recvbuf) + (0 * recvcount + seg * segment_count) * datatype_size);
      const int* b = (const int*)((rank == 1 ? (char*) sendbuf : (char*) recvbuf) + (1 * recvcount + seg * segment_count) * datatype_size);
      const int* c = (const int*)((rank == 2 ? (char*) sendbuf : (char*) recvbuf) + (2 * recvcount + seg * segment_count) * datatype_size);
      const int* d = (const int*)((rank == 3 ? (char*) sendbuf : (char*) recvbuf) + (3 * recvcount + seg * segment_count) * datatype_size);

      int* out = (int*)((char*)recvbuf + (((rank + 1) % 4) * recvcount + seg * segment_count) * datatype_size);

      // Kernel on stream for overlap
      sum4arrays<<<512, 512, 0, stream>>>(a, b, c, d, out, curr_seg_count);
  }

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  MPI_Waitall(next_send_req, send_req, MPI_STATUSES_IGNORE);

  free(send_req);
  free(recv_req);
  return MPI_SUCCESS;
}

int intra_allgather(void *recvbuf, size_t recvcount, MPI_Datatype recvtype,
                    MPI_Comm comm){
    // Do an allgather where each rank isends and irecvs from everyone else
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int datatype_size;
    MPI_Type_size(recvtype, &datatype_size);
    int next_send_req = 0, next_recv_req = 0;
    MPI_Request* send_req = (MPI_Request*) malloc(sizeof(MPI_Request) * size);
    MPI_Request* recv_req = (MPI_Request*) malloc(sizeof(MPI_Request) * size);
    for (int i = 0; i < size; i++) {
        if (i != rank) {
            MPI_Isend(((char*) recvbuf) + rank * recvcount * datatype_size, recvcount, recvtype, i, 0, comm, &send_req[next_send_req]);
            ++next_send_req;
            MPI_Irecv(((char*) recvbuf) + i * recvcount * datatype_size, recvcount, recvtype, i, 0, comm, &recv_req[next_recv_req]);
            ++next_recv_req;
        }
    }
    MPI_Waitall(next_recv_req, recv_req, MPI_STATUSES_IGNORE);
    MPI_Waitall(next_send_req, send_req, MPI_STATUSES_IGNORE);
    free(send_req);
    free(recv_req);
    return MPI_SUCCESS;
}

int VerifyCollective(int* buf_a, int* buf_b, int dim, int rank){
  int incorrect = 0;
  for(int i = 0; i<dim; ++i){
    try {
      if(buf_a[i] != buf_b[i]){
        cout << rank << " : "<< i <<" - swing: "<< buf_a[i] << " test: " << buf_b[i] << endl;
        incorrect = -1;
        return -1;
      }
    } catch (const invalid_argument& e) {
        cerr << "ERROR: Memory corruption on verification." << endl;
        return EXIT_FAILURE;
    }
  }
  return incorrect;
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
    
    int BENCHMARK_ITERATIONS = 100;
    if(argc >= 4){
      BENCHMARK_ITERATIONS = atoi(argv[3]);
    }

    size_t hier_segment_size = (size_count * multiplier_type)/(2*log_2(size));
    if(argc >= 5){
      hier_segment_size = atoi(argv[4]);
    }

    size_t inter_segment_size = (size_count * multiplier_type)/(2*log_2(size));
    if(argc >= 6){
      inter_segment_size = atoi(argv[5]);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    
#define GPUS_PER_NODE 4    
    int gpu_rank = rank % GPUS_PER_NODE;

    CUDA_CHECK(cudaSetDevice(gpu_rank));

    size_t BUFFER_SIZE = (size_count * multiplier_type);
    size_t msg_count = BUFFER_SIZE/sizeof(int);
    int *h_send_buffer = (int*) malloc(BUFFER_SIZE); 
    int *h_recv_buffer = (int*) malloc(BUFFER_SIZE);
    int *h_test_recv_buffer = (int*) malloc(BUFFER_SIZE);

    if(rank == 0){
      cerr << "BUFFER: " << size_count << size_type << ", GPUs: " << size << endl;
    }

    if(size_count == 512 && strcmp(size_type, "MiB") == 0){
      cout << " {" << rank << " : "<< processor_name << " - " << gpu_rank << "}" << endl;
    }

    int *d_send_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_send_buffer, (size_t) BUFFER_SIZE));
    int *d_recv_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_recv_buffer, (size_t) BUFFER_SIZE));
    int *d_test_recv_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_test_recv_buffer, (size_t) BUFFER_SIZE));
    srand(time(NULL)*rank);      
    for (size_t i = 0; i < msg_count; i++) {
        h_send_buffer[i] = rand()*rank % 10; 
    }

    // Create the inter and intra communicator
    MPI_Comm intra_comm, inter_comm;
    MPI_Comm_split(MPI_COMM_WORLD, (rank / GPUS_PER_NODE), rank, &intra_comm);
    MPI_Comm_split(MPI_COMM_WORLD, (rank % GPUS_PER_NODE), rank, &inter_comm);
    int intra_rank, inter_rank;
    int intra_size, inter_size;
    MPI_Comm_size(intra_comm, &intra_size);
    MPI_Comm_size(inter_comm, &inter_size);
    MPI_Comm_rank(intra_comm, &intra_rank);
    MPI_Comm_rank(inter_comm, &inter_rank);
    //printf("Rank %d - intra_rank %d - inter_rank %d - intra_size %d - inter_size %d\n", rank, intra_rank, inter_rank, intra_size, inter_size); fflush(stdout);


    CUDA_CHECK(cudaMemcpy(d_send_buffer, h_send_buffer, (size_t) BUFFER_SIZE, cudaMemcpyHostToDevice));
    
    // Hier allreduce
    // Do first a reduce-scatter on the intra communicator
    char* tmp_buf           = ((char*) d_recv_buffer) + ((intra_rank + 2) % GPUS_PER_NODE)*(msg_count / GPUS_PER_NODE)*sizeof(int);
    char* redscat_out_buf   = ((char*) d_recv_buffer) + ((intra_rank + 1) % GPUS_PER_NODE)*(msg_count / GPUS_PER_NODE)*sizeof(int);
    char* allreduce_out_buf = ((char*) d_recv_buffer) + intra_rank                        *(msg_count / GPUS_PER_NODE)*sizeof(int);

    // This is it
    //intra_reducescatter_block(d_send_buffer, d_recv_buffer, msg_count / GPUS_PER_NODE, MPI_INT, intra_comm);    
    intra_reducescatter_block_segmented(d_send_buffer, d_recv_buffer, msg_count / GPUS_PER_NODE, MPI_INT, intra_comm, hier_segment_size);    
    // d_recv_buffer is large enough, I can use part of it as recvbuf
    allreduce_swing_lat(redscat_out_buf, allreduce_out_buf, (msg_count / GPUS_PER_NODE), MPI_INT, MPI_SUM, inter_comm, tmp_buf);
    // Now I can do an allgather on the intra communicator
    intra_allgather(d_recv_buffer, (msg_count / GPUS_PER_NODE), MPI_INT, intra_comm);

    // Check allreduce
    MPI_Allreduce(d_send_buffer, d_test_recv_buffer, msg_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(h_recv_buffer, d_recv_buffer, (size_t) BUFFER_SIZE, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_test_recv_buffer, d_test_recv_buffer, (size_t) BUFFER_SIZE, cudaMemcpyDeviceToHost));
  
    ret = VerifyCollective(h_recv_buffer, h_test_recv_buffer, BUFFER_SIZE / sizeof(int), rank);
    if(ret==-1){
      cerr << "THE ANALYZED COLLECTIVE WITH " << BUFFER_SIZE << " IS NOT WORKING! :(" << endl;
      free(h_send_buffer);
      free(h_recv_buffer);
      free(h_test_recv_buffer);

      CUDA_CHECK(cudaFree(d_recv_buffer));
      CUDA_CHECK(cudaFree(d_send_buffer));
      CUDA_CHECK(cudaFree(d_test_recv_buffer));
      return EXIT_FAILURE;
    }    

    double* samples = (double*) malloc(sizeof(double) * BENCHMARK_ITERATIONS);
    double* samples_all = (double*) malloc(sizeof(double) * BENCHMARK_ITERATIONS);
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < BENCHMARK_ITERATIONS + WARM_UP; ++i){

        double start_time, end_time;
        start_time = MPI_Wtime();
        // This is it
        intra_reducescatter_block_segmented(d_send_buffer, d_recv_buffer, msg_count / GPUS_PER_NODE, MPI_INT, intra_comm, hier_segment_size);    
        // d_recv_buffer is large enough, I can use part of it as recvbuf
        allreduce_swing_lat(redscat_out_buf, allreduce_out_buf, (msg_count / GPUS_PER_NODE), MPI_INT, MPI_SUM, MPI_COMM_WORLD, tmp_buf);
        // Now I can do an allgather on the intra communicator
        intra_allgather(d_recv_buffer, msg_count / GPUS_PER_NODE, MPI_INT, intra_comm);
        end_time = MPI_Wtime();

        if(i>WARM_UP) {
          samples[i-WARM_UP] = (end_time - start_time)*1e9;
          total_time += (end_time - start_time);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    total_time = (double)(total_time)/BENCHMARK_ITERATIONS;

    double max_time;
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(samples, samples_all, BENCHMARK_ITERATIONS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(h_recv_buffer, d_recv_buffer, (size_t) BUFFER_SIZE, cudaMemcpyDeviceToHost));

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
      printf("highest\n");
      for(int i = 0; i < BENCHMARK_ITERATIONS; ++i){
        printf("%d\n", (int) samples[i]);
      }

      float buffer_gib = (BUFFER_SIZE / (float) (1024*1024*1024)) * 8;
      float bandwidth =  2 * buffer_gib * ((size-1)/(float)size);
      bandwidth = bandwidth / max_time;
    }

    if(rank == 0){
      cerr << "BUFFER: " << size_count << size_type << " DONE!" << endl;
    }

    free(h_send_buffer);
    free(h_recv_buffer);
    free(h_test_recv_buffer);

    CUDA_CHECK(cudaFree(d_recv_buffer));
    CUDA_CHECK(cudaFree(d_send_buffer));
    CUDA_CHECK(cudaFree(d_test_recv_buffer));
    MPI_Finalize();
    return EXIT_SUCCESS;
}

