#include <mpi.h>
#include <iostream>
#include <cstring>
#include <climits>
#include <cuda_runtime.h>

using namespace std;

#define MiB1 1048576
#define WARM_UP 10
#define BENCHMARK_ITERATIONS 100

// TO TEST THE FINAL VERSION
//https://github.com/NVIDIA/nccl-tests


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

  if((rank & 1) == 0) dest = (int)((rank + (1-1*pow(-2,step+1))/3) + comm_sz) % comm_sz;//(rank + rhos[step]) % comm_sz;  // Even rank
  else dest = (int)((rank - (1-1*pow(-2,step+1))/3) + comm_sz) % comm_sz; //(rank - rhos[step]) % comm_sz;                 // Odd rank

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

  int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_count = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
  int i, idx;
  for(i = 0; global_thread_idx + i*thread_count < count; i++){
    idx = global_thread_idx + i*thread_count; 
    inout[idx] += in[idx]; 
  }

  /*
  for(int i = 0; i<count; i++){
    inout[i] += in[i];
  }
  */
}

inline int log_2(int value) {
    if (1 > value) {
        return -1;
    }
    return sizeof(int)*8 - 1 - __builtin_clz(value);
}

int allreduce_swing_lat(const void *sbuf, void *rbuf, size_t count, MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) {
  
  int rank, size, datatype_size, ret;
  char *tmpsend, *tmprecv;
  char *inplacebuf_free; 
  ptrdiff_t extent, true_extent, lb, gap, span = 0;
  
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dtype, &datatype_size);

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

  tmpsend = inplacebuf;
  tmprecv = (char*) rbuf;
  
  // Determine nearest power of two less than or equal to size
  // and return an error if size is 0
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
      //reduce_sum_kernel<<<1, 1>>>((const int*)tmprecv, (int*)tmpsend, count);
      new_rank = rank >> 1;
    }
  } else new_rank = rank - extra_ranks;

  
  CUDA_CHECK(cudaDeviceSynchronize());

  // Actual allreduce computation for general cases
  // If the extra_ranks phasee has been done, this phase won't be executed. 
  // I'm running with 4 nodes, so the first phase won't be done but this yes.
  int s, vdest, dest;
  for(s = 0; s < steps; s++){
    if(loop_flag) break;
    vdest = pi(new_rank, s, adjsize);

    dest = is_power_of_two ?
              vdest :
              (vdest < extra_ranks) ?
              (vdest << 1) + 1 : vdest + extra_ranks;

    ret = MPI_Sendrecv(tmpsend, count, dtype, dest, 0,
                       tmprecv, count, dtype, dest, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != ret) { 
      cerr << "ERROR: AllGather phase." << endl;
      return -1;
    }
    
    reduce_sum_kernel<<<512, 512>>>((const int*)tmprecv, (int*)tmpsend, count);
    //reduce_sum_kernel<<<1, 1>>>((const int*)tmprecv, (int*)tmpsend, count);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Final results is sent to nodes that are not included in general computation
  // (general computation loop requires 2^n nodes).
  if(rank < (2 * extra_ranks)){
    if(!loop_flag){
      ret = MPI_Send(tmpsend, count, dtype, (rank - 1), 0, comm);
    } else {
      ret = MPI_Recv(rbuf, count, dtype, (rank + 1), 0, comm, MPI_STATUS_IGNORE);
      tmpsend = (char*)rbuf;
    }
  }

  if(tmpsend != rbuf) {
    CUDA_CHECK(cudaMemcpy(rbuf, tmpsend, count * datatype_size, cudaMemcpyDeviceToDevice));
  }

  CUDA_CHECK(cudaFree(inplacebuf_free));
  return MPI_SUCCESS;
}



int VerifyCollective(int* buf_a, int* buf_b, int dim, int rank){
  int incorrect = 0;
  for(int i = 0; i<dim; ++i){
    try {
      if(buf_a[i] != buf_b[i]){
        cout << rank << " : "<< i <<" - cuda: "<< buf_a[i] << " test: " << buf_b[i] << endl;
        incorrect = -1;
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

    if (argc < 2) {
        cerr << "Please, insert an integer as argument" << endl;
        return 1;  
    }

    int mib_count = 0;
    try {
      mib_count = stoi(argv[1]);  
      if(rank == 0)
        cout << endl << "Message is " << mib_count << " MiB - ALL REDUCE" << endl;
    } catch (const invalid_argument& e) {
        cout << "Not valid argument!" << endl;
        return EXIT_FAILURE;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    cout << " {" << rank << " : "<< processor_name << "}" << endl;

    int msg_count = (mib_count * MiB1)/sizeof(int);
    int BUFFER_SIZE = (mib_count * MiB1);
    int *h_send_buffer = (int*) malloc(BUFFER_SIZE); 
    int *h_recv_buffer = (int*) malloc(BUFFER_SIZE);
    int *h_test_recv_buffer = (int*) malloc(BUFFER_SIZE);

    // BISOGNA SETTARE LA GPU, FORSE BISOGNA FARLO SEMPRE
    int gpu_rank = rank % 1;
    CUDA_CHECK(cudaSetDevice(gpu_rank));

    int *d_send_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_send_buffer, (size_t) BUFFER_SIZE));
    int *d_recv_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_recv_buffer, (size_t) BUFFER_SIZE));
    int *d_test_recv_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_test_recv_buffer, (size_t) BUFFER_SIZE));

    for (int i = 0; i < msg_count; i++) {
        h_send_buffer[i] = (float) rank; 
    }
    CUDA_CHECK(cudaMemcpy(d_send_buffer, h_send_buffer, (size_t) BUFFER_SIZE, cudaMemcpyHostToDevice));

    allreduce_swing_lat(d_send_buffer, d_recv_buffer, (size_t) msg_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(d_send_buffer, d_test_recv_buffer, (size_t) msg_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(h_recv_buffer, d_recv_buffer, (size_t) BUFFER_SIZE, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_test_recv_buffer, d_test_recv_buffer, (size_t) BUFFER_SIZE, cudaMemcpyDeviceToHost));
  
    ret = VerifyCollective(h_recv_buffer, h_test_recv_buffer, BUFFER_SIZE/sizeof(int), rank);
    if(ret==-1){
      cerr << "THE ANALYZED COLLECTIVE IS NOT WORKING! :(" << endl;
      free(h_send_buffer);
      free(h_recv_buffer);
      free(h_test_recv_buffer);

      CUDA_CHECK(cudaFree(d_recv_buffer));
      CUDA_CHECK(cudaFree(d_send_buffer));
      CUDA_CHECK(cudaFree(d_test_recv_buffer));
      return EXIT_FAILURE;
    }


    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < BENCHMARK_ITERATIONS + WARM_UP; ++i){

        double start_time = MPI_Wtime();
        allreduce_swing_lat(d_send_buffer, d_recv_buffer, (size_t) msg_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        double end_time = MPI_Wtime();

        if(i>WARM_UP) {
            total_time += end_time - start_time;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    total_time = (double)(total_time)/BENCHMARK_ITERATIONS;

    double max_time;
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(h_recv_buffer, d_recv_buffer, (size_t) BUFFER_SIZE, cudaMemcpyDeviceToHost));

    int verifier = 0;
    for(int i = 0; i<msg_count; i++){
      verifier += h_recv_buffer[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
      float buffer_gib = (BUFFER_SIZE / (float) (1024*1024*1024)) * 8;
      float bandwidth =  buffer_gib * log_2(size);
      bandwidth = bandwidth / max_time;
      cout << "Buffer: "  << BUFFER_SIZE << " byte - " << buffer_gib << " Gib - " << mib_count << " MiB, verifier: " << verifier << ", Latency: " << max_time << ", Bandwidth: " << bandwidth << endl;
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

