#include <mpi.h>
#include <iostream>
#include <cstring>
#include <climits>
#include <cassert>
#include <cuda_runtime.h>
#include <nccl.h>
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
    
#define NCCL_CHECK(cmd) do {                        \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

typedef struct {
  uint* parent; // For each node in the tree, its parent.
  uint* reached_at_step; // For each node in the tree, the step at which it is reached.
  uint* remapped_ranks; // The remapped rank so that each subtree contains contiguous remapped ranks    
  uint* remapped_ranks_max; // remapped_ranks_max[i] is the maximum remapped rank in the subtree rooted at i
  uint* subtree_roots; // subtree_roots[i] is the rank of the root of the subtree to which i belongs
  // We do not need to store the min because it is the remapped rank itself (the node is the last in the subtree to be numbered)
  //uint* remapped_ranks_min; // remapped_ranks_min[i] is the minimum remapped rank in the subtree rooted at i
} swing_tree_t;

typedef enum {
  SWING_DISTANCE_INCREASING = 0,
  SWING_DISTANCE_DECREASING = 1
} swing_distance_type_t;


typedef enum {
  // Default
  SWING_ALGO_FAMILY_DEFAULT = 0,
  // Swing
  SWING_ALGO_FAMILY_SWING,
  // Recdoub
  SWING_ALGO_FAMILY_RECDOUB,
  // Bruck
  SWING_ALGO_FAMILY_BRUCK,
  // Ring
  SWING_ALGO_FAMILY_RING,
} swing_algo_family_t;

typedef struct swing_comm_info_key {
  uint root;
  uint port;
  swing_algo_family_t algo;
  swing_distance_type_t dist_type;
  MPI_Comm comm;  

  bool operator==(const swing_comm_info_key &other) const
  { return (root == other.root &&
            port == other.port &&
            algo == other.algo &&
            dist_type == other.dist_type &&
            comm == other.comm);
  }
} swing_comm_info_key_t;

template <>
struct std::hash<swing_comm_info_key_t>
{
  std::size_t operator()(const swing_comm_info_key_t& k) const
  {
    using std::size_t;
    using std::hash;
    using std::string;

    // Compute individual hash values for first,
    // second and third and combine them using XOR
    // and bit shifting:

    return (hash<uint>()(k.root) ^ 
            hash<uint>()(k.port) ^
            hash<uint>()(k.algo) ^
            hash<uint>()(k.dist_type) ^
            hash<void*>()((void*) k.comm));
  }
};

typedef struct {
  swing_tree_t tree;
} swing_comm_info_t;

std::unordered_map<swing_comm_info_key_t, swing_comm_info_t> comm_info;

typedef struct{
  uint d; // In which dimension is this global step performed
  uint step_in_d; // What's the relative step in this specific dimension
} swing_step_info_t;

class SwingCoordConverter {
  public:
      uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
      uint dimensions_num; 
      int* coordinates;
      uint size;
      uint num_steps_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
      uint num_steps;
  
      SwingCoordConverter(uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num);
      
      ~SwingCoordConverter();

      // Convert a rank id into a list of d-dimensional coordinates
      // Row-major order, i.e., row coordinates change the slowest 
      // (i.e., we first increase depth, than cols, then rows -- https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays) 
      // @param id (IN): the rank id
      // @param coord (OUT): the array where the coordinates are stored
      void getCoordFromId(int id, int* coord);

      // Convert d-dimensional coordinates into a rank id).
      // Dimensions are (rows, cols, depth).
      // @param coords (IN): the array with the coordinates
      // @return the rank id
      int getIdFromCoord(int* coords);

      // Gets the real or virtual (for non-p2) coordinates associated to a rank.
      // @param rank (IN): the rank
      // @param coord (OUT): the array where the coordinates are stored
      void retrieve_coord_mapping(uint rank, int* coord);
};

int ceil_log2(unsigned long long x){
  static const unsigned long long t[6] = {
    0xFFFFFFFF00000000ull,
    0x00000000FFFF0000ull,
    0x000000000000FF00ull,
    0x00000000000000F0ull,
    0x000000000000000Cull,
    0x0000000000000002ull
  };

  int y = (((x & (x - 1)) == 0) ? 0 : 1);
  int j = 32;
  int i;

  for (i = 0; i < 6; i++) {
    int k = (((x & t[i]) == 0) ? 0 : j);
    y += k;
    x >>= k;
    j >>= 1;
  }

  return y;
}

// Adapted from MPICH code -- https://github.com/pmodels/mpich/blob/94b1cd6f060cafbf68d6d83ea551a8bcc8fcecd4/src/mpi/topo/topo_impl.c
void SwingCoordConverter::getCoordFromId(int id, int* coord){
  int nnodes = 1;
  for(size_t i = 0; i < dimensions_num; i++){
      nnodes *= dimensions[i];
  }
  for (uint i = 0; i < dimensions_num; i++) {
      nnodes = nnodes / dimensions[i];
      coord[i] = id / nnodes;
      id = id % nnodes;
  }
}

// Adapted from MPICH code -- https://github.com/pmodels/mpich/blob/94b1cd6f060cafbf68d6d83ea551a8bcc8fcecd4/src/mpi/topo/topo_impl.c)
int SwingCoordConverter::getIdFromCoord(int* coords){
  int rank = 0;
  int multiplier = 1;
  int coord;
  for (int i = dimensions_num - 1; i >= 0; i--) {
      coord = coords[i];
      if (/*cart_ptr->topo.cart.periodic[i]*/ 1) {
          if (coord >= dimensions[i])
              coord = coord % dimensions[i]; 
          else if (coord < 0) {
              coord = coord % dimensions[i];
              if (coord)
                  coord = dimensions[i] + coord;
          }
      }
      rank += multiplier * coord;
      multiplier *= dimensions[i];
  }
  return rank;
}

// Cache coordinates
void SwingCoordConverter::retrieve_coord_mapping(uint rank, int* coord){
  if(coordinates[rank*dimensions_num] == -1){
      getCoordFromId(rank, &(coordinates[rank*dimensions_num]));
  }
  memcpy(coord, &(coordinates[rank*dimensions_num]), sizeof(uint)*dimensions_num);
}

SwingCoordConverter::SwingCoordConverter(uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num): dimensions_num(dimensions_num){
  memcpy(this->dimensions, dimensions, sizeof(uint)*dimensions_num);
  this->size = 1;
  this->num_steps = 0;
  for(size_t d = 0; d < dimensions_num; d++){
      this->size *= dimensions[d];
      this->num_steps_per_dim[d] = ceil_log2(dimensions[d]);
      this->num_steps += this->num_steps_per_dim[d];
  }
  this->coordinates = (int*) malloc(sizeof(int)*this->size*this->dimensions_num);
  memset(this->coordinates        , -1, sizeof(int)*this->size*dimensions_num);
}

SwingCoordConverter::~SwingCoordConverter(){
  free(this->coordinates);
}


static inline int is_odd(int x){
  return x & 1;
}

static int is_mirroring_port(int port, uint dimensions_num){
  if(dimensions_num == 3){
      return port >= dimensions_num;
  }else if(dimensions_num == 2){
      if(port == 0 || port == 1){
          return 0;
      }else if(port == 2 || port == 3){
          return 1;
      }else if(port == 4 || port == 5){
          // TODO: On 2D torus we might have some unbalance (i.e., 4 ports for plain collectives and 2 for mirrored) The data we sent on plain collectives is 2x higher than what we send on mirrored. We should unbalance the 6 partitions of the vector accordingly.
          return 0;
      }
  }else if(dimensions_num == 1){
      return port % 2;
  }
  return 0;
}

static int get_distance_sign(size_t rank, size_t port, size_t dimensions_num){
  int multiplier = 1;
  if(is_odd(rank)){ // Invert sign if odd rank
      multiplier *= -1;
  }
  if(is_mirroring_port(port, dimensions_num)){ // Invert sign if mirrored collective
      multiplier *= -1;     
  }
  return multiplier;
}

static inline int mod(int a, int b){
  int r = a % b;
  return r < 0 ? r + b : r;
}

void get_peer_c(int* coord_rank, size_t step, uint port, swing_step_info_t* step_info, swing_algo_family_t algo, uint dimensions_num, uint* dimensions, int* coord_peer){
  memcpy(coord_peer, coord_rank, sizeof(uint)*dimensions_num);
  size_t d = step_info[step].d;
  size_t step_in_d = step_info[step].step_in_d;
  if(algo == SWING_ALGO_FAMILY_RECDOUB){
      int distance = (coord_peer[d] ^ (1 << (step_in_d))) - coord_peer[d];
      if(is_mirroring_port(port, dimensions_num)){ // Invert sign if mirrored collective
          distance *= -1;     
      }
      coord_peer[d] = mod(coord_peer[d] + distance, dimensions[d]);
  }else if(algo == SWING_ALGO_FAMILY_SWING){
      int distance = rhos[step_in_d];
      distance *= get_distance_sign(coord_rank[d], port, dimensions_num);
      coord_peer[d] = mod(coord_peer[d] + distance, dimensions[d]);
  }else{
      fprintf(stderr, "Unknown algorithm family\n");
      exit(EXIT_FAILURE);
  }
}

inline int log_2(int value) {
  if (1 > value) {
      return -1;
  }
  return sizeof(int)*8 - 1 - __builtin_clz(value);
}

/*
static inline uint32_t reverse(uint32_t x){
  x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
  x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
  x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
  x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
  x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
  return x;
}

static uint32_t binary_to_negabinary(int32_t bin) {
  assert(bin <= 0x55555555);
  const uint32_t mask = 0xAAAAAAAA;
  return (mask + bin) ^ mask;
}

static inline int in_range(int x, uint32_t nbits){
  return x >= smallest_negabinary[nbits] && x <= largest_negabinary[nbits];
}

static inline uint32_t get_rank_negabinary_representation(uint32_t num_ranks, uint32_t rank){
  binary_to_negabinary(rank);
  uint32_t nba = UINT32_MAX, nbb = UINT32_MAX;
  size_t num_bits = log_2(num_ranks);
  if(rank % 2){
      if(in_range(rank, num_bits)){
          nba = binary_to_negabinary(rank);
      }
      if(in_range(rank - num_ranks, num_bits)){
          nbb = binary_to_negabinary(rank - num_ranks);
      }
  }else{
      if(in_range(-rank, num_bits)){
          nba = binary_to_negabinary(-rank);
      }
      if(in_range(-rank + num_ranks, num_bits)){
          nbb = binary_to_negabinary(-rank + num_ranks);
      }
  }

  assert(nba != UINT32_MAX || nbb != UINT32_MAX);

  if(nba == UINT32_MAX && nbb != UINT32_MAX){
      return nbb;
  }else if(nba != UINT32_MAX && nbb == UINT32_MAX){
      return nba;
  }else{ // Check MSB
      if(nba & (80000000 >> (32 - num_bits))){
          return nba;
      }else{
          return nbb;
      }
  }
}
*/

static void remap_ranks(int* coord_root, size_t step, uint port, swing_algo_family_t algo, swing_distance_type_t dist_type, SwingCoordConverter* scc, uint* next_rank, const uint* parent, uint* remapped_ranks, uint* remapped_ranks_max, swing_step_info_t* step_info){
  remapped_ranks_max[scc->getIdFromCoord(coord_root)] = *next_rank;
  for(size_t i = step; i < scc->num_steps; i++){
      int peer_rank[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
      int real_step;
      if(dist_type == SWING_DISTANCE_DECREASING){
          real_step = scc->num_steps - 1 - i;
      }else{
          real_step = i;
      }
      get_peer_c(coord_root, real_step, port, step_info, algo, scc->dimensions_num, scc->dimensions, peer_rank);

      // I need to check if I am actually the parent of that peer.
      // When we have a number of nodes that is not a power of 2, we may have peers which are reached by
      // more than one node, so we must do this check.
      if(parent[scc->getIdFromCoord(peer_rank)] == scc->getIdFromCoord(coord_root)){
          remap_ranks(peer_rank, i + 1, port, algo, dist_type, scc, next_rank, parent, remapped_ranks, remapped_ranks_max, step_info);
      }
  }
  remapped_ranks[scc->getIdFromCoord(coord_root)] = (*next_rank);
  //DPRINTF("Remapped rank %d to %d\n", scc->getIdFromCoord(coord_root), *next_rank);
  (*next_rank)--;
}

swing_step_info_t* compute_step_info(uint port, SwingCoordConverter* scc, uint dimensions_num, uint* dimensions){
  size_t next_step_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
  size_t current_d = port % dimensions_num;
  swing_step_info_t* step_info = (swing_step_info_t*) malloc(sizeof(swing_step_info_t)*scc->num_steps);
  memset(next_step_per_dim, 0, sizeof(size_t)*dimensions_num);
  for(size_t i = 0; i < scc->num_steps; i++){
      step_info[i].d = current_d;
      step_info[i].step_in_d = next_step_per_dim[current_d];

      // Move to the next dimension for the next step
      size_t d = current_d;              
      // Increase the next step, unless we are done with this dimension
      if(next_step_per_dim[d] < ceil_log2(dimensions[d])){ 
          next_step_per_dim[d] += 1;
      }
      
      // Select next dimension
      if(i != scc->num_steps - 1){
          do{ 
              current_d = (current_d + 1) % dimensions_num;
              d = current_d;
          }while(next_step_per_dim[d] >= ceil_log2(dimensions[d])); // If we exhausted this dimension, move to the next one
      }
  }
  return step_info;
}

static void build_tree(int* coord_root, size_t step, uint port, swing_algo_family_t algo, swing_distance_type_t dist_type, SwingCoordConverter* scc, uint32_t* reached_at_step, uint32_t subtree_root, uint32_t* parent, swing_step_info_t* step_info, uint32_t* subtree_roots){
  for(size_t i = step; i < scc->num_steps; i++){
      int peer_rank[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
      int real_step;
      if(dist_type == SWING_DISTANCE_DECREASING){
          real_step = scc->num_steps - 1 - i;
      }else{
          real_step = i;
      }        
      get_peer_c(coord_root, real_step, port, step_info, algo, scc->dimensions_num, scc->dimensions, peer_rank);
      
      uint32_t rank = scc->getIdFromCoord(peer_rank);
      if(parent[rank] == UINT32_MAX || i < reached_at_step[rank]){
          parent[rank] = scc->getIdFromCoord(coord_root);
          reached_at_step[rank] = i;
          if(step == 0){
              // If this is a children of the actual root, it is rooted in itself
              subtree_roots[rank] = rank;
          }else{
              subtree_roots[rank] = subtree_root;
          }
      }
      uint32_t actual_subtree_root = subtree_root;
      if(step == 0){
          // If I am actually the root, I can change the subroot_rank
          actual_subtree_root = rank;
      }
      build_tree(peer_rank, i + 1, port, algo, dist_type, scc, reached_at_step, actual_subtree_root, parent, step_info, subtree_roots);
  }
}

swing_tree_t get_tree(uint root, uint port, swing_algo_family_t algo, swing_distance_type_t dist_type, SwingCoordConverter* scc){
  swing_comm_info_key_t key;
  key.root = root;
  key.port = port;
  key.algo = algo;
  key.dist_type = dist_type;
  if(comm_info.find(key) != comm_info.end()){
      return comm_info[key].tree;
  }else{
      int coord_root[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
      scc->getCoordFromId(root, coord_root);
      swing_tree_t tree;
      uint* buffer = (uint*) malloc(sizeof(uint)*scc->size*5); // Do one single malloc rather than 5
      tree.parent = buffer;
      tree.reached_at_step = buffer + scc->size;
      tree.remapped_ranks = buffer + scc->size*2;
      tree.remapped_ranks_max = buffer + scc->size*3;
      tree.subtree_roots = buffer + scc->size*4;
      for(size_t i = 0; i < scc->size; i++){
          tree.parent[i] = UINT32_MAX;
          tree.reached_at_step[i] = scc->num_steps;
      }
             
      // Compute the basic tree informations (parent and reached_at_step)
      swing_step_info_t* step_info = compute_step_info(port, scc, scc->dimensions_num, scc->dimensions);
      build_tree(coord_root, 0, port, algo, dist_type, scc, tree.reached_at_step, 0, tree.parent, step_info, tree.subtree_roots);    
      tree.parent[root] = UINT32_MAX;
      tree.reached_at_step[root] = 0; // To avoid sending the step for myself at a wrong value
      tree.subtree_roots[root] = UINT32_MAX;

      // Now that we have a loopless tree, do a DFS to compute the remapped rank
      uint next_rank = scc->size - 1;
      remap_ranks(coord_root, 0, port, algo, dist_type, scc, &(next_rank), tree.parent, tree.remapped_ranks, tree.remapped_ranks_max, step_info);
      assert(next_rank == UINT32_MAX);
      free(step_info);
      swing_comm_info_t cinfo;
      cinfo.tree = tree;
#pragma omp critical
      {
      comm_info[key] = cinfo;
      }
      return tree;
  }
}

void compute_peers(uint rank, int port, swing_algo_family_t algo, SwingCoordConverter* scc, uint* peers){
  bool terminated_dimensions_bitmap[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
  uint8_t next_step_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
  memset(next_step_per_dim, 0, sizeof(uint8_t)*LIBSWING_MAX_SUPPORTED_DIMENSIONS);
  
  // Compute default directions
  int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
  scc->retrieve_coord_mapping(rank, coord);
  for(size_t i = 0; i < scc->dimensions_num; i++){
      terminated_dimensions_bitmap[i] = false;            
  }
  
  int target_dim, relative_step, distance, last_dim = port - 1;
  uint terminated_dimensions = 0, o = 0;
  
  // Generate peers
  for(size_t i = 0; i < (uint) scc->num_steps; ){            
      if(scc->dimensions_num > 1){
          scc->retrieve_coord_mapping(rank, coord); // Regenerate rank coord
          o = 0;
          do{
              target_dim = (last_dim + 1 + o) % (scc->dimensions_num);            
              o++;
          }while(terminated_dimensions_bitmap[target_dim]);
          relative_step = next_step_per_dim[target_dim];
          ++next_step_per_dim[target_dim];
          last_dim = target_dim;
      }else{
          target_dim = 0;
          relative_step = i;
          coord[0] = rank;
      }

      if(algo == SWING_ALGO_FAMILY_RECDOUB){
          distance = (coord[target_dim] ^ (1 << relative_step)) - coord[target_dim];
      }else if(algo == SWING_ALGO_FAMILY_SWING){
          distance = rhos[relative_step];
          // Flip the sign for odd nodes
          if(is_odd(coord[target_dim])){distance *= -1;}
      }else{
          fprintf(stderr, "Unknown algorithm family\n");
          exit(EXIT_FAILURE);
      }
      
      // Mirrored collectives
      if(is_mirroring_port(port, scc->dimensions_num)){distance *= -1;}

      if(relative_step < scc->num_steps_per_dim[target_dim]){
          coord[target_dim] = mod((coord[target_dim] + distance), scc->dimensions[target_dim]); // We need to use mod to avoid negative coordinates
          if(scc->dimensions_num > 1){
              peers[i] = scc->getIdFromCoord(coord);
          }else{
              peers[i] = coord[0];
          }
  
          i += 1;
      }else{
          terminated_dimensions_bitmap[target_dim] = true;
          terminated_dimensions++;                
      }        
  }        
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


double allreduce_swing_bdw_mesh(const void *send_buf, void *recv_buf, size_t count,
  ncclDataType_t dtype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, uint *peers, swing_tree_t *tree){

  int size, rank, dest, steps, step, datatype_size;
  int *r_count = NULL, *s_count = NULL, *r_index = NULL, *s_index = NULL;
  size_t w_size;
  uint32_t vrank, vdest;
  double start, total_time = 0.0;

  char *tmp_send = NULL, *tmp_recv = NULL;
  char *tmp_buf_raw = NULL, *tmp_buf;
  ptrdiff_t buf_size;

  ncclCommCount(comm, &size);
  ncclCommUserRank(comm, &rank);
  datatype_size = sizeof(datatype_size); // Convert to bits

  // Does not support non-power-of-two or negative sizes
  steps = log_2(size);

  // Allocate temporary buffer for send/recv and reduce operations
  buf_size = datatype_size * (count >> 1);
  
  CUDA_CHECK(cudaMalloc((void**) &tmp_buf_raw, buf_size));
  tmp_buf = tmp_buf_raw;

  // Copy into receive_buffer content of send_buffer to not produce
  // side effects on send_buffer
  if(send_buf != MPI_IN_PLACE) {
    CUDA_CHECK(cudaMemcpy(recv_buf, send_buf, count * datatype_size, cudaMemcpyDeviceToDevice));
  }
  
  
  r_index = (int*) malloc(sizeof(*r_index) * steps);
  s_index = (int*) malloc(sizeof(*s_index) * steps);
  r_count = (int*) malloc(sizeof(*r_count) * steps);
  s_count = (int*) malloc(sizeof(*s_count) * steps);

  w_size = count;
  s_index[0] = r_index[0] = 0;
  vrank = tree->remapped_ranks[rank];

  //Reduce-Scatter phase
  for(step = 0; step < steps; step++) {
    
    dest = peers[step];
    vdest = tree->remapped_ranks[dest];

    if(vrank < vdest) {
      r_count[step] = w_size / 2;
      s_count[step] = w_size - r_count[step];
      s_index[step] = r_index[step] + r_count[step];
    } else {
      s_count[step] = w_size / 2;
      r_count[step] = w_size - s_count[step];
      r_index[step] = s_index[step] + s_count[step];
    }

    tmp_send = (char *)recv_buf + s_index[step] * datatype_size;
    
    ncclGroupStart();
    ncclSend(tmp_send, s_count[step], dtype, dest, comm, stream);
    ncclRecv(tmp_buf, r_count[step], dtype, dest, comm, stream);
    ncclGroupEnd();

    tmp_recv = (char *) recv_buf + r_index[step] * datatype_size;

    reduce_sum_kernel<<<512, 512>>>((const int*)tmp_buf, (int*)tmp_recv, r_count[step]);

    if(step + 1 < steps) {
      r_index[step + 1] = r_index[step];
      s_index[step + 1] = r_index[step];
      w_size = r_count[step];
    }
    //printf("REDSCAT Rank: %d, Step: %d, Dest: %d, VRank: %u, VDest: %u, TIME: %lf \n", rank, step, dest, vrank, vdest, MPI_Wtime()-start);
  }

  // CUDA_CHECK(cudaDeviceSynchronize());

  // Allgather phase
  for(step = steps - 1; step >= 0; step--) {
    
    dest = peers[step];

    tmp_send = (char *)recv_buf + r_index[step] * datatype_size;
    tmp_recv = (char *)recv_buf + s_index[step] * datatype_size;

    ncclGroupStart();
    ncclSend(tmp_send, r_count[step], dtype, dest, comm, stream);
    ncclRecv(tmp_recv, s_count[step], dtype, dest, comm, stream);
    ncclGroupEnd();
    //printf("ALLGATH Rank: %d, Step: %d, Dest: %d, VRank: %u, VDest: %u, TIME: %lf \n", rank, step, dest, vrank, vdest, MPI_Wtime()-start);
  }

  CUDA_CHECK(cudaFree(tmp_buf_raw));
  free(r_index);
  free(s_index);
  free(r_count);
  free(s_count);
  return total_time;
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

    // Initialize NCCL
    ncclUniqueId id;
    if (rank == 0)
        NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    int gpu_rank = 0;
    if(rank < 2 * (size/4)){
      if(rank%2 == 0){
        gpu_rank = 0;
      } else {
        gpu_rank = 1;
      }
    } else {
      if(rank%2 == 0){
        gpu_rank = 2;
      } else {
        gpu_rank = 3 ;
      }
    }

    if(size == 4){
      gpu_rank = rank;
    }

    CUDA_CHECK(cudaSetDevice(gpu_rank));

    if(size_count == 512 && strcmp(size_type, "B") == 0){
      cout << " {" << rank << " : "<< processor_name << " - " << gpu_rank << "}" << endl;
    }

    size_t BUFFER_SIZE = (size_count * multiplier_type);
    long long int msg_count = BUFFER_SIZE/sizeof(int);
    int *h_send_buffer = (int*) malloc(BUFFER_SIZE); 
    int *h_recv_buffer = (int*) malloc(BUFFER_SIZE);
    int *h_test_recv_buffer = (int*) malloc(BUFFER_SIZE);

    int *d_send_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_send_buffer, (size_t) BUFFER_SIZE));
    int *d_recv_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_recv_buffer, (size_t) BUFFER_SIZE));
    int *d_test_recv_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_test_recv_buffer, (size_t) BUFFER_SIZE));


    SwingCoordConverter* scc = new SwingCoordConverter(new uint[2]{2, (uint) size/2}, 2);
    uint* peers = (uint*) malloc(sizeof(uint)*scc->size); 
    swing_tree_t tree = get_tree(0, 0, SWING_ALGO_FAMILY_SWING, SWING_DISTANCE_INCREASING, scc);
    compute_peers(rank, 0, SWING_ALGO_FAMILY_SWING, scc, peers);
    
    for (int i = 0; i < msg_count; i++) {
        h_send_buffer[i] = rank; 
    }
    CUDA_CHECK(cudaMemcpy(d_send_buffer, h_send_buffer, (size_t) BUFFER_SIZE, cudaMemcpyHostToDevice));


    // NCCL stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    //CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); //BHO?

    ncclComm_t comm;
    //ncclResult_t state;
    NCCL_CHECK(ncclCommInitRank(&comm, size, id, rank));
    //ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    //config.blocking = 0;
    //config.minCTAs = 4;
    //config.maxCTAs = 16;
    //config.cgaClusterSize = 2;
    //config.netName = "Socket";
    //CUDACHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking))


    allreduce_swing_bdw_mesh(d_send_buffer, d_recv_buffer, (size_t) msg_count, ncclInt, ncclSum, comm, stream, peers, &tree);
    ncclAllReduce(d_send_buffer, d_test_recv_buffer, (size_t) msg_count, ncclInt, ncclSum, comm, stream);

    CUDA_CHECK(cudaMemcpy(h_recv_buffer, d_recv_buffer, (size_t) BUFFER_SIZE, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_test_recv_buffer, d_test_recv_buffer, (size_t) BUFFER_SIZE, cudaMemcpyDeviceToHost));
  
    ret = 0;//VerifyCollective(h_recv_buffer, h_test_recv_buffer, BUFFER_SIZE/sizeof(int), rank);
    if(ret==-1){
      cerr << "THE ANALYZED COLLECTIVE IS NOT WORKING! :(" << endl;
      free(h_send_buffer);
      free(h_recv_buffer);
      free(h_test_recv_buffer);
      free(peers);

      CUDA_CHECK(cudaFree(d_recv_buffer));
      CUDA_CHECK(cudaFree(d_send_buffer));
      CUDA_CHECK(cudaFree(d_test_recv_buffer));
      return EXIT_FAILURE;
    }

    double* samples = (double*) malloc(sizeof(double) * BENCHMARK_ITERATIONS);
    double* samples_all = (double*) malloc(sizeof(double) * BENCHMARK_ITERATIONS);
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < BENCHMARK_ITERATIONS + WARM_UP; ++i){

        double time_to_remove = 0.0;
        double start_time = MPI_Wtime();
        ncclAllReduce(d_send_buffer, d_test_recv_buffer, (size_t) msg_count, ncclInt, ncclSum, comm, stream);
        //time_to_remove = allreduce_swing_bdw_mesh(d_send_buffer, d_recv_buffer, (size_t) msg_count, ncclInt, ncclSum, comm, stream, peers, &tree);
        time_to_remove = 0;
        double end_time = MPI_Wtime();
        if(i>WARM_UP) {
            samples[i-WARM_UP] = (end_time - start_time);
            total_time += (end_time - start_time) - time_to_remove;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    total_time = (double)(total_time)/BENCHMARK_ITERATIONS;

    double max_time;
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(samples, samples_all, BENCHMARK_ITERATIONS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(h_recv_buffer, d_recv_buffer, (size_t) BUFFER_SIZE, cudaMemcpyDeviceToHost));

    uint64_t verifier = 0;
    for(int i = 0; i<msg_count; i++){
      verifier += h_recv_buffer[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
      cout << "highest" << endl;
      for(int i = 0; i < BENCHMARK_ITERATIONS; ++i){
        cout << samples_all[i] << endl;
      }

      float buffer_gib = (BUFFER_SIZE / (float) (1024*1024*1024)) * 8;
      float bandwidth =  2 * buffer_gib * ((size-1)/(float)size);
      bandwidth = bandwidth / max_time;
      cout << "-> Buffer: "  << BUFFER_SIZE << " byte - " << buffer_gib << " Gib - " << size_count << size_type << ", verifier: " << verifier << ", Latency: " << max_time << ", Bandwidth: " << bandwidth << endl;
    }

    free(h_send_buffer);
    free(h_recv_buffer);
    free(h_test_recv_buffer);
    free(peers);

    CUDA_CHECK(cudaFree(d_recv_buffer));
    CUDA_CHECK(cudaFree(d_send_buffer));
    CUDA_CHECK(cudaFree(d_test_recv_buffer));

    MPI_Finalize();
    return EXIT_SUCCESS;
}

