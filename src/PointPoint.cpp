#include "common.hpp"

using namespace std;

#define B1 1
#define KiB1 1024
#define MiB1 1048576
#define GiB1 1073741824


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size, name_len, ret;
    double total_time = 0.0;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(processor_name, &name_len);

    if(size != 2){
      cerr << "Point-Point requires only 2 procs!" << endl;
      return 1;
    }

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
    int WARM_UP = 10;
    if(argc >= 5){
      WARM_UP = atoi(argv[4]);
    }


    MPI_Barrier(MPI_COMM_WORLD);
  

    int BUFFER_SIZE = (size_count * multiplier_type);
    int msg_count = BUFFER_SIZE/sizeof(int);
    int *send_buffer = (int*) malloc_align(BUFFER_SIZE*size);
    int *recv_buffer = (int*) malloc_align(BUFFER_SIZE*size);
    if (send_buffer == NULL || recv_buffer == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    for (int i = 0; i < msg_count; i++) {
        send_buffer[i] = (int) (rand()*rank % 10);
    }

    double* samples = (double*) malloc_align(sizeof(double) * BENCHMARK_ITERATIONS);
    double* samples_all = (double*) malloc_align(sizeof(double) * BENCHMARK_ITERATIONS);
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < BENCHMARK_ITERATIONS + WARM_UP; ++i){

        double start_time, end_time;
        start_time = MPI_Wtime();
        if(rank == 0){
          if(i % 2 == 0){
              MPI_Send(send_buffer, msg_count, MPI_INT, 1, 0, MPI_COMM_WORLD);
              MPI_Recv(recv_buffer, msg_count, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }else{
              MPI_Recv(recv_buffer, msg_count, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              MPI_Send(send_buffer, msg_count, MPI_INT, 1, 0, MPI_COMM_WORLD);
          }
        }

        if(rank == 1){
          if(i % 2 == 0){
            MPI_Recv(recv_buffer, msg_count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(send_buffer, msg_count, MPI_INT, 0, 0, MPI_COMM_WORLD);
          } else {
            MPI_Send(send_buffer, msg_count, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(recv_buffer, msg_count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        }
        end_time = MPI_Wtime();

        if(i>WARM_UP) {
          samples[i-WARM_UP] = (end_time - start_time)/2.0;
          total_time += (end_time - start_time);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    total_time = (double)(total_time)/BENCHMARK_ITERATIONS;

    double max_time;
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(samples, samples_all, BENCHMARK_ITERATIONS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
      printf("highest\n");
      for(int i = 0; i < BENCHMARK_ITERATIONS; ++i){
        printf("%.9f\n", samples_all[i]);
      }
    }

    if(rank == 0){
      cerr << "BUFFER: " << size_count << size_type << " DONE!" << endl;
    }

    free(send_buffer);
    free(recv_buffer);
    free(samples);
    free(samples_all);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
