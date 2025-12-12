rm -rf build
mkdir build

if [ -z "$1" ]; then
    echo "Error: Missing argument. Please provide a target (e.g., haicgu, leonardo, cresco8 or test)."
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

if [ "$1" == "test" ]; then
    echo "Testing compilation environment..."
fi

# mpicxx src/ReduceScatter.cpp -O3 -o build/reducescatter
# mpicxx src/AllReduce_raw.cpp -O3 -o build/allreduce
# mpicxx src/AllGather.cpp -O3 -o build/allgather
# mpicxx src/All2All.cpp -O3 -o build/all2all
# mpicxx src/PointPoint.cpp -O3 -o build/pointpoint
# mpicxx src/PointPoint_async.cpp -O3 -o build/pointpoint_async
mpicxx src/All2All_raw.cpp -O3 -o build/all2all_raw
mpicxx src/AllGather_raw.cpp -O3 -o build/allgather_raw

#CONGESTION NOISE
mpicxx src/NoiseAll2All.cpp -O3 -o build/noise_all2all
mpicxx src/NoiseIncast.cpp -O3 -o build/noise_incast

#BURSTS
mpicxx src/All2All_raw_burst.cpp -O3 -o build/all2all_raw_burst

