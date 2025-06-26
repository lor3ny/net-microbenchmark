rm -rf build
mkdir build

GPU_SUFFIX=""
GPU_FNAME=""
MPICC=""
GPUCC=""
LIBS=""
INCLUDES=""
if [ "$1" == "marenostrum5" ]; then
    module purge  
    module load EB/apps EB/install OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.2.0 nccl
    GPU_SUFFIX="cu"
    GPU_FNAME="CUDA"
    MPICC="mpicxx -O3"
    GPUCC="nvcc -O3"
    LIBS="-lcudart -L${OPENMPI_LIB}"
    INCLUDES="-I${OPENMPI_INCLUDE}"
elif [ "$1" == "leonardo" ]; then
    module purge
    module load nccl
    module load openmpi/4.1.6--nvhpc--24.3
    GPU_SUFFIX="cu"
    GPU_FNAME="CUDA"
    MPICC="mpicxx -O3"
    GPUCC="nvcc -O3"
    LIBS="-lcudart -L${OPENMPI_LIB}"
    INCLUDES="-I${OPENMPI_INCLUDE}"
elif [ "$1" == "lumi" ]; then
    module purge
    module load PrgEnv-cray
    module load LUMI/24.03 partition/G
    module load aws-ofi-rccl
    module load craype-accel-amd-gfx90a
    module load rocm
    GPU_SUFFIX="hip"
    GPU_FNAME="HIP"
    MPICC="CC -xhip --offload-arch=gfx90a -O3"
    #GPUCC="hipcc" # --offload-arch=gfx90a"
    GPUCC=${MPICC}
    LIBS="-L${HIP_LIB_PATH} -L${MPICH_DIR}/lib -xhip"
    INCLUDES="-I${MPICH_DIR}/include"
else
    echo "Please specify system name"
    exit
fi

${MPICC} src/gpu/AllReduceHIER_BW_MPI.${GPU_SUFFIX} -O3 -o build/allreduce_hier_bw_mpi ${LIBS}
${MPICC} src/gpu/AllReduceHIER_LAT_MPI.${GPU_SUFFIX} -O3 -o build/allreduce_hier_lat_mpi ${LIBS}
if [ "$1" != "lumi" ]; then
    nvcc src/gpu/AllReduceNCCL.${GPU_SUFFIX} -O3 ${INCLUDES} -o build/allreduce_nccl ${LIBS} -lnccl -lmpi
    #nvcc src/gpu/AllReduceHIER_BW_NCCL.${GPU_SUFFIX} -O3 -o build/allreduce_hier_bw_nccl -lmpi -lnccl
fi
${GPUCC} src/gpu/AllReduce${GPU_FNAME}-AWARE.cpp -O3 ${INCLUDES} -o build/allreduce_cudaaware ${LIBS} -lmpi
