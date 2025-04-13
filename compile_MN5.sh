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
#export LD_LIBRARY_PATH=/apps/ACC/EASYBUILD/software/NVHPC/23.7-CUDA-12.2.0/Linux_x86_64/23.7/cuda/12.2/lib64:$LD_LIBRARY_PATH
#export PATH=/apps/ACC/EASYBUILD/software/binutils/2.40-GCCcore-12.3.0/bin/:$PATH
module purge 
module load EB/apps EB/install OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.2.0
mpicxx src/gpu/AllReduceHIER_BW_MPI.cu -O3 -o build/allreduce_hier_bw_mpi -lcudart
mpicxx src/gpu/AllReduceHIER_LAT_MPI.cu -O3 -o build/allreduce_hier_lat_mpi -lcudart

module purge
module load EB/apps EB/install OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.2.0
module load nccl
nvcc src/gpu/AllReduceHIER_BW_NCCL.cu -O3 -o build/allreduce_hier_bw_nccl -lmpi -lnccl
nvcc src/gpu/AllReduceNCCL.cu -O3 -o build/allreduce_nccl -lmpi -lnccl
mpicxx src/gpu/AllReduceCUDA-AWARE.cpp -O3 -o build/allreduce_cudaaware -lcudart