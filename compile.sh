rm -rf build
mkdir build
mpicxx src/ReduceScatter.cpp -O3 -o build/reducescatter
mpicxx src/AllReduce.cpp -O3 -o build/allreduce
mpicxx src/AllGather.cpp -O3 -o build/allgather
mpicxx src/AllReduceOPNULL.cpp -O3 -o build/allreduceOPNULL
mpicxx src/ReduceScatterOPNULL.cpp -O3 -o build/reducescatterOPNULL
