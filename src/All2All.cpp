#include "common.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size, name_len, ret;
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
    int WARM_UP = 10;
    if(argc >= 5){
      WARM_UP = atoi(argv[4]);
    }


    MPI_Barrier(MPI_COMM_WORLD);
  

    int BUFFER_SIZE = (size_count * multiplier_type);
    unsigned char *send_buffer = (unsigned char*) malloc_align(BUFFER_SIZE*size); 
    unsigned char *recv_buffer = (unsigned char*) malloc_align(BUFFER_SIZE*size);
    if (send_buffer == NULL || recv_buffer == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    for (int i = 0; i < BUFFER_SIZE*size; i++) {
        send_buffer[i] = 'a'; 
    }

    double* samples = (double*) malloc_align(sizeof(double) * BENCHMARK_ITERATIONS);
    double* samples_all = (double*) malloc_align(sizeof(double) * BENCHMARK_ITERATIONS * size);
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < BENCHMARK_ITERATIONS + WARM_UP; ++i){

        double start_time, end_time;
        start_time = MPI_Wtime();
        MPI_Alltoall(send_buffer, BUFFER_SIZE, MPI_BYTE, recv_buffer, BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);
        end_time = MPI_Wtime();

        if(i>WARM_UP) {
          samples[i-WARM_UP] = (end_time - start_time);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    double max_time;
    //MPI_Reduce(samples, samples_all, BENCHMARK_ITERATIONS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Gather(samples, BENCHMARK_ITERATIONS, MPI_DOUBLE, samples_all, BENCHMARK_ITERATIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
      printf("highest\n");
      int times_count = sizeof(samples_all)/sizeof(double);
      for(int i = 0; i < times_count; ++i){
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
