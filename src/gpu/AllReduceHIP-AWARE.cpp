#include <mpi.h>
#include <iostream>
#include <cstring>
#include <climits>
#include <cassert>
#include <hip/hip_runtime.h>
#include <unordered_map>
#include <omp.h>

using namespace std;

#define B1 1
#define KiB1 1024
#define MiB1 1048576
#define GiB1 1073741824
#define WARM_UP 10

#define CUDA_CHECK(cmd) do {                        \
  hipError_t e = cmd;                              \
  if( e != hipSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,hipGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


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
    MPI_Barrier(MPI_COMM_WORLD);
    
    int gpu_rank = rank%4;
    CUDA_CHECK(hipSetDevice(gpu_rank));

    if(size_count == 512 && strcmp(size_type, "B") == 0){
      cout << " {" << rank << " : "<< processor_name << " - " << gpu_rank << "}" << endl;
    }

    size_t BUFFER_SIZE = (size_count * multiplier_type);
    long long int msg_count = BUFFER_SIZE/sizeof(int);
    int *h_send_buffer = (int*) malloc(BUFFER_SIZE); 
    int *h_recv_buffer = (int*) malloc(BUFFER_SIZE);
    int *h_test_recv_buffer = (int*) malloc(BUFFER_SIZE);

    int *d_send_buffer;
    CUDA_CHECK(hipMalloc((void**)&d_send_buffer, (size_t) BUFFER_SIZE));
    int *d_recv_buffer;
    CUDA_CHECK(hipMalloc((void**)&d_recv_buffer, (size_t) BUFFER_SIZE));
    int *d_test_recv_buffer;
    CUDA_CHECK(hipMalloc((void**)&d_test_recv_buffer, (size_t) BUFFER_SIZE));

    
    for (int i = 0; i < msg_count; i++) {
        h_send_buffer[i] = rank; 
    }
    CUDA_CHECK(hipMemcpy(d_send_buffer, h_send_buffer, (size_t) BUFFER_SIZE, hipMemcpyHostToDevice));

    MPI_Allreduce(d_send_buffer, d_recv_buffer, (size_t) msg_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(d_send_buffer, d_test_recv_buffer, (size_t) msg_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    CUDA_CHECK(hipMemcpy(h_recv_buffer, d_recv_buffer, (size_t) BUFFER_SIZE, hipMemcpyDeviceToHost));
    CUDA_CHECK(hipMemcpy(h_test_recv_buffer, d_test_recv_buffer, (size_t) BUFFER_SIZE, hipMemcpyDeviceToHost));
  
    ret = VerifyCollective(h_recv_buffer, h_test_recv_buffer, BUFFER_SIZE/sizeof(int), rank);
    if(ret==-1){
      cerr << "THE ANALYZED COLLECTIVE IS NOT WORKING! :(" << endl;
      free(h_send_buffer);
      free(h_recv_buffer);
      free(h_test_recv_buffer);

      CUDA_CHECK(hipFree(d_recv_buffer));
      CUDA_CHECK(hipFree(d_send_buffer));
      CUDA_CHECK(hipFree(d_test_recv_buffer));
      return EXIT_FAILURE;
    }

    double* samples = (double*) malloc(sizeof(double) * BENCHMARK_ITERATIONS);
    double* samples_all = (double*) malloc(sizeof(double) * BENCHMARK_ITERATIONS);
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < BENCHMARK_ITERATIONS + WARM_UP; ++i){

        double start_time = MPI_Wtime();
        MPI_Allreduce(d_send_buffer, d_recv_buffer, (size_t) msg_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        double end_time = MPI_Wtime();

        if(i>WARM_UP) {
            samples[i-WARM_UP] = (end_time - start_time)*1e9;
            total_time += end_time - start_time;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    total_time = (double)(total_time)/BENCHMARK_ITERATIONS;

    double max_time;
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(samples, samples_all, BENCHMARK_ITERATIONS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(hipMemcpy(h_recv_buffer, d_recv_buffer, (size_t) BUFFER_SIZE, hipMemcpyDeviceToHost));

    uint64_t verifier = 0;
    for(int i = 0; i<msg_count; i++){
      verifier += h_recv_buffer[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
      printf("highest\n");
      for(int i = 0; i < BENCHMARK_ITERATIONS; ++i){
        printf("%d\n", (int) samples[i]);
      }

      float buffer_gib = (BUFFER_SIZE / (float) (1024*1024*1024)) * 8;
      float bandwidth =  2 * buffer_gib * ((size-1)/(float)size);
      bandwidth = bandwidth / max_time;
      cout << "Buffer: "  << BUFFER_SIZE << " byte - " << buffer_gib << " Gib - " << size_count << size_type << ", verifier: " << verifier << ", Latency: " << max_time << ", Bandwidth: " << bandwidth << endl;
    }

    free(h_send_buffer);
    free(h_recv_buffer);
    free(h_test_recv_buffer);

    CUDA_CHECK(hipFree(d_recv_buffer));
    CUDA_CHECK(hipFree(d_send_buffer));
    CUDA_CHECK(hipFree(d_test_recv_buffer));

    MPI_Finalize();
    return EXIT_SUCCESS;
}

