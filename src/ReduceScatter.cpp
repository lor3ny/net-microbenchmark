#include <mpi.h>
#include <iostream>
#include <cstring>

using namespace std;

#define MiB1 1048576
#define WARM_UP 10
#define BENCHMARK_ITERATIONS 100


void noop(void *in, void *inout, int *len, MPI_Datatype *datatype) {
  return;
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size, name_len;
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
        cout << endl << "Message is " << mib_count << " MiB - REDUCE SCATTER" << endl;
    } catch (const invalid_argument& e) {
        cout << "Not valid argument!" << endl;
        return EXIT_FAILURE;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    cout << " {" << rank << " : "<< processor_name << "}" << endl;

    int msg_count = (mib_count * MiB1)/sizeof(float);
    int BUFFER_SIZE = (mib_count * MiB1);
    int DATA_COUNT = (BUFFER_SIZE / sizeof(int));
    float *send_buffer = (float*) malloc(BUFFER_SIZE); 
    float *recv_buffer = (float*) malloc(BUFFER_SIZE/size);
    int *recvcounts = (int*)malloc(size * sizeof(int));
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

    MPI_Op noop_op;
    MPI_Op_create((MPI_User_function *)noop, 1, &noop_op);

    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < BENCHMARK_ITERATIONS + WARM_UP; ++i){

        double start_time = MPI_Wtime();
        MPI_Reduce_scatter(send_buffer, recv_buffer, recvcounts, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        double end_time = MPI_Wtime();

        if(i>WARM_UP) {
          total_time += end_time - start_time;
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
        cout << "Buffer: " << BUFFER_SIZE << " byte - " << buffer_gib << " Gib - " << mib_count << " MiB, verifier: " << verifier << ", Latency: " << total_time << ", Bandwidth: " << bandwidth << endl;
    }

    free(send_buffer);
    free(recv_buffer);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

