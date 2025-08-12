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

static inline int copy_buffer_different_dt (const void *input_buffer, size_t scount,
                                            const MPI_Datatype sdtype, void *output_buffer,
                                            size_t rcount, const MPI_Datatype rdtype) {
  if (input_buffer == NULL || output_buffer == NULL || scount <= 0 || rcount <= 0) {
    return MPI_ERR_UNKNOWN;
  }

  int sdtype_size;
  MPI_Type_size(sdtype, &sdtype_size);
  int rdtype_size;
  MPI_Type_size(rdtype, &rdtype_size);

  size_t s_size = (size_t) sdtype_size * scount;
  size_t r_size = (size_t) rdtype_size * rcount;

  if (r_size < s_size) {
    memcpy(output_buffer, input_buffer, r_size); // Copy as much as possible
    return MPI_ERR_TRUNCATE;      // Indicate truncation
  }

  memcpy(output_buffer, input_buffer, s_size);        // Perform the memory copy

  return MPI_SUCCESS;
}


void allgather_memcpy(const void *sbuf, size_t scount, MPI_Datatype sdtype, void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm){

  int rank, size, sendto, recvfrom, i, recvdatafrom, senddatafrom;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  MPI_Type_get_extent(rdtype, &rlb, &rext);

  tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
  if (MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
  }
}


void allgather_ring(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                   void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm) {

  int rank, size, sendto, recvfrom, i, recvdatafrom, senddatafrom;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  MPI_Type_get_extent(rdtype, &rlb, &rext);

  sendto = (rank + 1) % size;
  recvfrom  = (rank - 1 + size) % size;

  for (i = 0; i < size - 1; i++) {

    recvdatafrom = (rank - i - 1 + size) % size;
    senddatafrom = (rank - i + size) % size;

    tmprecv = (char*)rbuf + (ptrdiff_t)recvdatafrom * (ptrdiff_t)rcount * rext;
    tmpsend = (char*)rbuf + (ptrdiff_t)senddatafrom * (ptrdiff_t)rcount * rext;

    MPI_Sendrecv(tmpsend, rcount, rdtype, sendto, 0,
                       tmprecv, rcount, rdtype, recvfrom, 0,
                       comm, MPI_STATUS_IGNORE);

  }
}

/*
giusto un double check sul calcolo della banda. Supponendo che il parametro bufsize di MPI è N e che la collettiva è runnata su P nodi:
-per la allreduce la banda è calcolata come 2*N*((P-1)/P) / tempo
-per la alltoall è calcolata come N*(P-1) / tempo
-per la allgather come N*((P-1)/P) / tempo
*/

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


    MPI_Barrier(MPI_COMM_WORLD);


    int BUFFER_SIZE = (size_count * multiplier_type);
    int MSG_BUFFER_SIZE = BUFFER_SIZE / size;
    unsigned char *send_buffer = (unsigned char*) malloc_align(MSG_BUFFER_SIZE);
    unsigned char *recv_buffer = (unsigned char*) malloc_align(BUFFER_SIZE);
    if (send_buffer == NULL || recv_buffer == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    srand(time(NULL)*rank); 
    for (int i = 0; i < MSG_BUFFER_SIZE; i++) {
        send_buffer[i] = (int) (rand()*rank % 10);
    }

    // TESTING THE COLLECTIVE
 
    unsigned char *t_recv_buffer = (unsigned char*) malloc_align(BUFFER_SIZE*size);
    allgather_memcpy(send_buffer, MSG_BUFFER_SIZE, MPI_BYTE, recv_buffer, MSG_BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);
    allgather_ring(send_buffer, MSG_BUFFER_SIZE, MPI_BYTE, recv_buffer, MSG_BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);

    MPI_Allgather(send_buffer, MSG_BUFFER_SIZE, MPI_BYTE, t_recv_buffer, MSG_BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);

    if(VerifyCollective(recv_buffer, t_recv_buffer, BUFFER_SIZE, rank) == -1){
      cerr << "ERROR[" << rank << "]: Custom collective didn't pass the validation!" << endl;
      return EXIT_FAILURE;
    } else {
      cerr << "Test passed!" << endl;
    }

    // TESTING THE COLLECTIVE

    double* samples = (double*) malloc_align(sizeof(double) * BENCHMARK_ITERATIONS);
    double* samples_all = (double*) malloc_align(sizeof(double) * BENCHMARK_ITERATIONS);
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < BENCHMARK_ITERATIONS + WARM_UP; ++i){

        double start_time, end_time;
        start_time = MPI_Wtime();
        allgather_memcpy(send_buffer, MSG_BUFFER_SIZE, MPI_BYTE, recv_buffer, MSG_BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);
        allgather_ring(send_buffer, MSG_BUFFER_SIZE, MPI_BYTE, recv_buffer, MSG_BUFFER_SIZE, MPI_BYTE, MPI_COMM_WORLD);
        end_time = MPI_Wtime();

        if(i>WARM_UP) {
          samples[i-WARM_UP] = (end_time - start_time);
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
