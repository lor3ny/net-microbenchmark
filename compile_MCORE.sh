rm -rf build
mkdir build

if [ "$1" == "haicgu" ]; then
    module purge 
    module load GCC
    module load OpenMPI
fi

if [ "$1" == "leonardo" ]; then
    module purge 
    module openpmi
fi

mpicxx src/ReduceScatter.cpp -O3 -o build/reducescatter
mpicxx src/AllReduce.cpp -O3 -o build/allreduce
mpicxx src/AllGather.cpp -O3 -o build/allgather
mpicxx src/ReduceScatter_noop.cpp -O3 -o build/reducescatter_noop
mpicxx src/All2All.cpp -O3 -o build/all2all
