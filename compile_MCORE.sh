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
mpicxx src/PointPoint.cpp -O3 -o build/pointpoint
mpicxx src/PointPoint_async.cpp -O3 -o build/pointpoint_async
mpicxx src/All2All_Raw.cpp -O3 -o build/all2all_raw
mpicxx src/AllGather_Raw.cpp -O3 -o build/allgather_raw
mpicxx src/NoiseAll2All.cpp -O3 -o build/noise_all2all
mpicxx src/NoiseIncast.cpp -O3 -o build/noise_incast

