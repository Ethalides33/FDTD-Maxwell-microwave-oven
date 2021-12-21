#!/bin/bash

#SBATCH --job-name=mpi-job
#SBATCH --time=01:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000

module load releases/2020b
module load OpenMPI/4.1.0-GCC-10.2.0
module load Silo/4.11-foss-2020b

cd $SLURM_SUBMIT_DIR

mkdir r
mpirun -np $SLURM_NTASKS ./microwave ./params.txt
