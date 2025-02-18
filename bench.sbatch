#!/bin/bash
#SBATCH --partition=cn-eth
#SBATCH --output=log_eth/%j.out
#SBATCH --error=log_eth/%j.err
#SBATCH --ntasks=4
#SBATCH --nodes=4 
#SBATCH --ntasks-per-node=1           
#SBATCH --time=00:10:00         
#SBATCH --exclusive      

module load GCC
module load OpenMPI

#export OMPI_MCA_base_verbose=100
#export OMPI_MCA_btl_base_verbose=100

#export OMPI_MCA_coll_base_verbose=1
export OMPI_MCA_coll_tuned_use_dynamic_rules=1
export OMPI_MCA_coll_tuned_allgather_algorithm=4
export OMPI_MCA_coll_tuned_allgather_algorithm_segmentsize=4096
export OMPI_MCA_coll_tuned_allreduce_algorithm=4
export OMPI_MCA_coll_tuned_allreduce_algorithm_segmentsize=4096
#export OMPI_MCA_coll_tuned_allreduce_algorithm_tree_fanout=4
#export OMPI_MCA_coll_tuned_dynamic_rules_filename="/home/lpiarulli/rules.txt"

#mpirun --rankfile rankfile.txt ./allgather 2
#mpirun --rankfile rankfile.txt ./allgather 4
#mpirun --rankfile rankfile.txt ./allgather 16

#mpirun --rankfile rankfile.txt ./allreduce 2
#mpirun --rankfile rankfile.txt ./allreduce 4
#mpirun --rankfile rankfile.txt ./allreduce 16

srun ./allgather 2
srun ./allgather 4
srun ./allgather 16

srun ./allreduce 2
srun ./allreduce 4
srun ./allreduce 16

#--mca coll_base_verbose 1
#ompi_info --param coll tuned --level 9      to see available algorithms

# Set the mpirun flags to change the algorithm of all reduce
#--mca coll_tuned_use_dynamic_rules 1 \
#--mca coll_tuned_allreduce_algorithm 3 \
#--mca coll_tuned_allreduce_algorithm_segmentsize 4096 \
#--mca coll_tuned_allreduce_algorithm_tree_fanout 4
