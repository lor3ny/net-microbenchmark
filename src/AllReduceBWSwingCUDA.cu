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


inline int log_2(int value) {
  if (1 > value) {
      return -1;
  }
  return sizeof(int)*8 - 1 - __builtin_clz(value);
}


inline uint32_t remap_rank(uint32_t num_ranks, uint32_t rank){
  // NEGABINARY COMPUTATION IS DONE BY SAVERIO, CHECK IF IT IS NECESSARY
  uint32_t remap_rank = get_rank_negabinary_representation(num_ranks, rank);    
  remap_rank = remap_rank ^ (remap_rank >> 1);
  size_t num_bits = log_2(num_ranks);
  remap_rank = reverse(remap_rank) >> (32 - num_bits);
  return remap_rank;
}


__global__ void reduce_sum_kernel(const int *in, int *inout, size_t count) {

  int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_count = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
  int i, idx;
  for(i = 0; global_thread_idx + i*thread_count < count; i++){
    idx = global_thread_idx + i*thread_count; 
    inout[idx] += in[idx]; 
  }
}

int allreduce_swing_bdw_remap(const void *send_buf, void *recv_buf, size_t count,
  MPI_Datatype dtype, MPI_Op op, MPI_Comm comm){
  int size, rank, dest, steps, step, err = MPI_SUCCESS;
  int *r_count = NULL, *s_count = NULL, *r_index = NULL, *s_index = NULL;
  size_t w_size;
  uint32_t vrank, vdest;

  char *tmp_send = NULL, *tmp_recv = NULL;
  char *tmp_buf_raw = NULL, *tmp_buf;
  ptrdiff_t lb, extent, true_extent, gap = 0, buf_size;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Does not support non-power-of-two or negative sizes
  steps = log_2(size);

  // Allocate temporary buffer for send/recv and reduce operations
  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_get_true_extent(dtype, &gap, &true_extent);
  buf_size = true_extent + extent * (count >> 1);
  tmp_buf_raw = (char *)malloc(buf_size);
  tmp_buf = tmp_buf_raw - gap;

  // Copy into receive_buffer content of send_buffer to not produce
  // side effects on send_buffer
  if(send_buf != MPI_IN_PLACE) {

    // FARE UNA MEMCPY

    err = copy_buffer((char *)send_buf, (char *)recv_buf, count, dtype);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
  }


  CUDA_CHECK(cudaMalloc((void**) &r_index, sizeof(*r_index) * steps);
  CUDA_CHECK(cudaMalloc((void**) &s_index, sizeof(*s_index) * steps);
  CUDA_CHECK(cudaMalloc((void**) &r_count, sizeof(*r_count) * steps);
  CUDA_CHECK(cudaMalloc((void**) &s_count, sizeof(*s_count) * steps);
  if(NULL == r_index || NULL == s_index || NULL == r_count || NULL == s_count) {
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }

  w_size = count;
  s_index[0] = r_index[0] = 0;
  vrank = remap_rank((uint32_t) size, (uint32_t) rank);

  // Reduce-Scatter phase
  for(step = 0; step < steps; step++) {
    dest = pi(rank, step, size);
    vdest = remap_rank((uint32_t) size, (uint32_t) dest);

    //QUESTA ROBA NON SI PUÒ FARE, COME EVITARLA?
    if(vrank < vdest) {
      r_count[step] = w_size / 2;
      s_count[step] = w_size - r_count[step];
      s_index[step] = r_index[step] + r_count[step];
    } else {
      s_count[step] = w_size / 2;
      r_count[step] = w_size - s_count[step];
      r_index[step] = s_index[step] + s_count[step];
    }
    tmp_send = (char *)recv_buf + s_index[step] * extent;
    err = MPI_Sendrecv(tmp_send, s_count[step], dtype, dest, 0,
    tmp_buf, r_count[step], dtype, dest, 0,
    comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }

    tmp_recv = (char *) recv_buf + r_index[step] * extent;
    MPI_Reduce_local(tmp_buf, tmp_recv, r_count[step], dtype, op);

    if(step + 1 < steps) {
      r_index[step + 1] = r_index[step];
      s_index[step + 1] = r_index[step];
      w_size = r_count[step];
    }
  }

  // Allgather phase
  for(step = steps - 1; step >= 0; step--) {
    dest = pi(rank, step, size);

    tmp_send = (char *)recv_buf + r_index[step] * extent;
    tmp_recv = (char *)recv_buf + s_index[step] * extent;
    err = MPI_Sendrecv(tmp_send, r_count[step], dtype, dest, 0,
    tmp_recv, s_count[step], dtype, dest, 0,
    comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
  }

  free(tmp_buf_raw);
  free(r_index);
  free(s_index);
  free(r_count);
  free(s_count);
  return MPI_SUCCESS;

  cleanup_and_return:
  if(NULL != tmp_buf_raw)  free(tmp_buf_raw);
  if(NULL != r_index)      free(r_index);
  if(NULL != s_index)      free(s_index);
  if(NULL != r_count)      free(r_count);
  if(NULL != s_count)      free(s_count);
  return err;
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
    int gpu_rank = rank % 4;
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

    allreduce_swing_bdw_remap(d_send_buffer, d_recv_buffer, (size_t) msg_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
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
        allreduce_swing_bdw_remap(d_send_buffer, d_recv_buffer, (size_t) msg_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
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

