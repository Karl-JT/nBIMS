#!/bin/bash

#SBATCH -J singularity_test
#SBATCH -o singularity_test.out
#SBATCH -e singularity_test.err
#SBATCH --time=24:00:00
#SBATCH --partition=hsw_v100_32g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=yjuntao@nvidia.com
#SBATCH --mail-type=ALL

unset XDG_RUNTIME_DIR

module unload xalt/2.6.3-gpu
module load singularity/3.2.0
module load PrgEnv/GCC+MPICH/2016-07-22 
module load mpich/3.1.3

#mpirun singularity exec --nv /datasets/yjuntao/images/petsc_cuda.simg ./testSolve -vec_type mpi -mat_type mpiaij
#mpirun singularity exec --nv /datasets/yjuntao/images/petsc_cuda.simg ./testSolve -vec_type seqcuda -mat_type seqaijcusparse

mpirun singularity exec --nv /datasets/yjuntao/images/petsc_cuda.simg ./testSolve -fieldsplit_0_ksp_type gmres -fieldsplit_0_pc_type gamg
