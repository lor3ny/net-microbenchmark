rm -rf build
mkdir build

if [ "$1" == "Marenostrum" ]; then
    module purge  
    module load EB/apps EB/install OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.2.0 nccl
fi

if [ "$1" == "leonardo" ]; then
    module purge
    module load nccl
    module load openmpi/4.1.6--nvhpc--24.3
fi

mpicxx src/gpu/AllReduceHIER_BW_MPI.cu -O3 -o build/allreduce_hier_bw_mpi -lcudart
mpicxx src/gpu/AllReduceHIER_LAT_MPI.cu -O3 -o build/allreduce_hier_lat_mpi -lcudart
nvcc src/gpu/AllReduceCUDA-AWARE.cpp -O3 -I${OPENMPI_INCLUDE} -o build/allreduce_cudaaware -lcudart  -lnccl -L${OPENMPI_LIB} -lmpi
nvcc src/gpu/AllReduceNCCL.cu -O3 -I${OPENMPI_INCLUDE} -o build/allreduce_nccl -lcudart -lnccl -L${OPENMPI_LIB} -lmpi
#nvcc src/gpu/AllReduceHIER_BW_NCCL.cu -O3 -o build/allreduce_hier_bw_nccl -lmpi -lnccl
