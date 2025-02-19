#include <mpi.h>
#include <iostream>

using namespace std;

#define MiB1 1048576 
#define WARM_UP 10
#define BENCHMARK_ITERATIONS 100

//#define BUFFER_SIZE (16 * MiB1 / sizeof(float)) DEBUG PURPOSE

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
    float *send_buffer = (float*) malloc(BUFFER_SIZE*size); 
    float *recv_buffer = (float*) malloc(BUFFER_SIZE*size);
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
        MPI_Alltoall(send_buffer, BUFFER_SIZE, MPI_BYTE, recv_buffer, BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);
        double end_time = MPI_Wtime();

        if(i>WARM_UP) {
            total_time += end_time - start_time;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    total_time = (total_time)/ (double) BENCHMARK_ITERATIONS;

    double max_time;
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    float verifier = 0;
    for(int i = 0; i<msg_count*size; i++){
        verifier += recv_buffer[i];
    }


    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
        float buffer_gib = (BUFFER_SIZE / (float) (1024*1024*1024)) * 8;
        float bandwidth =  buffer_gib * size;
        bandwidth = bandwidth / max_time;
        cout << rank << " : "<< processor_name <<" : ALL GATHER -> Buffer size (byte): " << BUFFER_SIZE << " - " << mib_count << " MiB, verifier: " << verifier << ", Time: " << total_time << ", Bandwidth: " << bandwidth << endl;
    }

    free(send_buffer);
    free(recv_buffer);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

