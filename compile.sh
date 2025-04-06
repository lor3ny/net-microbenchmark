rm -rf build
mkdir build


#mpicxx src/ReduceScatter.cpp -O3 -o build/reducescatter
#mpicxx src/AllReduce.cpp -O3 -o build/allreduce
#mpicxx src/AllGather.cpp -O3 -o build/allgather
#mpicxx src/ReduceScatter_noop.cpp -O3 -o build/reducescatter_noop
#mpicxx src/All2All.cpp -O3 -o build/all2all
#mpicxx src/test.c -O3 -o build/test
#nvcc src/gpu/AllReduceLATSwingMPI.cu -O3 -o build/allreduce_swing_mesh_mpi -lcudart -lcuda -lmpi
#nvcc src/gpu/AllReduceBWSwingMPI.cu -O3 -o build/allreduce_swing_mesh_mpi -lcudart -lcuda -lmpi
#mpicxx src/gpu/AllReduceMESHSwingMPI.cu -O3 -o build/allreduce_swing_mesh_mpi -lcudart

module purge 
module load openmpi/4.1.6--nvhpc--24.3 cuda
mpicxx src/gpu/AllReduceHIERSwingMPI.cu -O3 -o build/allreduce_swing_hier_mpi -lcudart
mpicxx src/gpu/AllReduceCUDA-AWARE.cpp -O3 -o build/allreduce_cudaaware -lcudart

module purge
module load openmpi cuda nccl
nvcc src/gpu/AllReduceNCCL.cu -O3 -o build/allreduce_nccl -lmpi -lnccl

#mpicxx src/gpu/AllGatherCUDA-AWARE.cpp -O3 -o build/allgather_cudaaware -lcudart
