rm -rf build
mkdir build

if [ -z "$1" ]; then
    echo "Error: Missing argument. Please provide a target (e.g., haicgu or leonardo)."
    exit 1
fi

if [ "$1" == "haicgu" ]; then
    module purge 
    module load GCC
    module load OpenMPI
fi

if [ "$1" == "leonardo" ]; then
    module purge 
    module load openmpi
fi

mpicxx src/ReduceScatter.cpp -O3 -o build/reducescatter
mpicxx src/AllReduce.cpp -O3 -o build/allreduce
mpicxx src/AllGather.cpp -O3 -o build/allgather
mpicxx src/All2All.cpp -O3 -o build/all2all
mpicxx src/PingPong.cpp -O3 -o build/pingpong
