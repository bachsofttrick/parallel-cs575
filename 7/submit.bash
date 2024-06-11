#!/bin/bash
# Define variables for number of nodes, total MPI tasks, and number of MPI processes per node
NUM_NODES=8
TOTAL_MPI_TASKS=8
MPI_PROCESSES_PER_NODE=4

#SBATCH -J AutoCorrelate
#SBATCH -A cs475-575
#SBATCH -p classmpitest
#SBATCH -N 8 # number of nodes
#SBATCH -n 8 # number of tasks
#SBATCH -o mpiproject.out
#SBATCH -e mpiproject.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=phanx@oregonstate.edu
module load openmpi
mpic++ mpiproject.cpp -o mpiproject -lm
mpiexec -mca btl self,tcp -np $MPI_PROCESSES_PER_NODE ./mpiproject
