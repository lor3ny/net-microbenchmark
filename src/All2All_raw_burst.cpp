#include "common.hpp"

using namespace std;

int VerifyCollective(unsigned char* buf_a, unsigned char* buf_b, int dim, int rank){
  int incorrect = 0;
  for(int i = 0; i<dim; ++i){
    try {
      if(buf_a[i] != buf_b[i]){
        incorrect = -1;
      }
    } catch (const invalid_argument& e) {
        cerr << "ERROR: Memory corruption on verification." << endl;
        return EXIT_FAILURE;
    }
  }
  return incorrect;
}

void all2all_memcpy(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm){

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int datatype_size;
    MPI_Type_size(sendtype, &datatype_size);

    const char* sbuf = static_cast<const char*>(sendbuf);
    char* rbuf = static_cast<char*>(recvbuf);

    double mem_time = MPI_Wtime(); 
    // Copy local data directly (self-send)
    std::memcpy(rbuf + rank * datatype_size * recvcount,
                sbuf + rank * datatype_size * sendcount,
                sendcount * datatype_size);

}

void custom_alltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                     void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int datatype_size;
    MPI_Type_size(sendtype, &datatype_size);

    const char* sbuf = static_cast<const char*>(sendbuf);
    char* rbuf = static_cast<char*>(recvbuf);

    // double mem_time = MPI_Wtime(); 
    // // Copy local data directly (self-send)
    // std::memcpy(rbuf + rank * datatype_size * recvcount,
    //             sbuf + rank * datatype_size * sendcount,
    //             sendcount * datatype_size);

    // double final_mem_time = MPI_Wtime() - mem_time;
    // cerr << "MEM: " << final_mem_time << "s" << endl;
    
    // double comm_time = MPI_Wtime();
    std::vector<MPI_Request> requests;
    for (int i = 0; i < size; ++i) {
        if (i == rank) continue;

        MPI_Request req_recv;
        MPI_Request req_send;

        MPI_Isend(sbuf + i * datatype_size * sendcount, sendcount, sendtype, i, 0, comm, &req_send);
        MPI_Irecv(rbuf + i * datatype_size * recvcount, recvcount, recvtype, i, 0, comm, &req_recv);
        
        requests.push_back(req_send);
        requests.push_back(req_recv);
    }

    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    // double final_comm_time = MPI_Wtime() - comm_time;
    // cerr << "COMM: " << final_comm_time << "s" << endl;
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
    int WARM_UP = 10;
    if(argc >= 5){
      WARM_UP = atoi(argv[4]);
    }
    double burst_pause=1e-6;
    if(argc >= 6){
      burst_pause = atof(argv[5]);
    }
    double burst_length=1e-2*3; //30ms
    if(argc >= 7){
      burst_length = atof(argv[6]);
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

    srand(time(NULL)*rank); 
    for (int i = 0; i < BUFFER_SIZE*size; i++) {
        send_buffer[i] = rand()*rank % size; 
    }

    // TESTING THE COLLECTIVE
 
    unsigned char *t_recv_buffer = (unsigned char*) malloc_align(BUFFER_SIZE*size);
    all2all_memcpy(send_buffer, BUFFER_SIZE, MPI_BYTE, recv_buffer, BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);
    custom_alltoall(send_buffer, BUFFER_SIZE, MPI_BYTE, recv_buffer, BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);

    MPI_Alltoall(send_buffer, BUFFER_SIZE, MPI_BYTE, t_recv_buffer, BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);

    if(VerifyCollective(recv_buffer, t_recv_buffer, BUFFER_SIZE*size, rank) == -1){
      cerr << "ERROR[" << rank << "]: Custom collective didn't pass the validation!" << endl;
      return EXIT_FAILURE;
    } else {
      if(rank == 0){
        cerr << "Test passed! Benchmark Proceed." << endl;  
      }
    }

    // TESTING THE COLLECTIVE
    vector<double> samples;

    MPI_Barrier(MPI_COMM_WORLD);

    bool burst_pause_rand = false;

    double burst_start_time;
    double measure_start_time;
    double burst_length_mean=burst_length;
    double burst_pause_mean=burst_pause;
    int burst_cont=0;
    int curr_iters=0;

    for(int i=0; i<BENCHMARK_ITERATIONS + WARM_UP; i++){
      
      if(burst_pause_rand){ /*randomized break length*/
          burst_pause=rand_expo(burst_pause_mean);
      }
      curr_iters=0;

      burst_start_time=MPI_Wtime();
      do{
          MPI_Barrier(MPI_COMM_WORLD);

          double start_time, end_time;
          all2all_memcpy(send_buffer, BUFFER_SIZE, MPI_BYTE, recv_buffer, BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);
          start_time = MPI_Wtime();
          custom_alltoall(send_buffer, BUFFER_SIZE, MPI_BYTE, recv_buffer, BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);
          end_time = MPI_Wtime();

          if(i>WARM_UP) {
            samples.push_back(end_time - start_time);
          }

          MPI_Barrier(MPI_COMM_WORLD);

          curr_iters++;
          if(burst_length!=0){ /*bcast needed for synch if bursts timed*/
              if(rank == 0){ /*master decides if burst should be continued*/
                  burst_cont=((MPI_Wtime()-burst_start_time)<burst_length);
              }
              MPI_Bcast(&burst_cont,1, MPI_INT, 0, MPI_COMM_WORLD); /*bcast the masters decision*/
          }

      }while(burst_cont);

      if(burst_pause!=0){
          if(burst_pause_rand){ /*randomized break length*/
              burst_pause=rand_expo(burst_pause_mean);
          }
          dsleep(burst_pause);
      }
    }

    vector<double> samples_all = vector<double>(samples.size(), 0.0);
    MPI_Reduce(samples.data(), samples_all.data(), samples.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
      printf("highest");
      for(int i = 0; i < samples_all.size(); ++i){
        printf("%.9f\n", samples_all[i]);
      }
    }

    if(rank == 0){
      cerr << "BUFFER: " << size_count << size_type <<" DONE! Bursts of " << curr_iters << endl;
    }

    free(send_buffer);
    free(recv_buffer);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
