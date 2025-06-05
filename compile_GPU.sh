rm -rf build
mkdir build

if [ "$1" == "Marenostrum" ]; then
    module purge  
    module load EB/apps EB/install OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.2.0 nccl
fi

if [ "$1" == "leonardo" ]; then
    module purge
    module load nccl
    module load openmpi/4.1.6--nvhpc--23.11
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib/
fi

mpicxx src/gpu/AllReduceHIER_BW_MPI.cu -O3 -o build/allreduce_hier_bw_mpi -lcudart
mpicxx src/gpu/AllReduceHIER_LAT_MPI.cu -O3 -o build/allreduce_hier_lat_mpi -lcudart
#nvcc src/gpu/AllReduceHIER_BW_NCCL.cu -O3 -o build/allreduce_hier_bw_nccl -lmpi -lnccl
nvcc -I/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/include -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib src/gpu/AllReduceNCCL.cu -O3 -o build/allreduce_nccl -lcudart -lnccl -lmpi
#mpicxx src/gpu/AllReduceCUDA-AWARE.cpp -O3 -o build/allreduce_cudaaware -lcudart
