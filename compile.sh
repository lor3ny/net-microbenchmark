#rm -rf build
#mkdir build
#mpicxx src/ReduceScatter.cpp -O3 -o build/reducescatter
#mpicxx src/AllReduce.cpp -O3 -o build/allreduce
#mpicxx src/AllGather.cpp -O3 -o build/allgather
#mpicxx src/AllReduceOPNULL.cpp -O3 -o build/allreduceOPNULL
#mpicxx src/ReduceScatterOPNULL.cpp -O3 -o build/reducescatterOPNULL
#mpicxx src/All2All.cpp -O3 -o build/all2all
#nvcc src/AllReduceSwingCUDA.cu -O3 -o build/allreduceSwingCUDA -lcudart -lcuda -lmpi
#nvcc src/AllReduceBWSwingCUDA.cu -O3 -o build/allreduceBWSwingCUDA -lcudart -lcuda -lmpi
#nvcc src/AllReduceMeshSwingCUDA.cu -O3 -o build/allreduceMeshSwingCUDA -lcudart -lcuda -lmpi
nvcc src/AllReduceMeshSwingNCCL.cu -O3 -o build/allreduceMeshSwingNCCL -lnccl -lcudart -lcuda -lmpi
