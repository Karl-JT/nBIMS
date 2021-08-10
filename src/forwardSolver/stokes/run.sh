#!/bin/bash

#SBATCH --nodes=2
#SBATCH --partition=hsw_v100_32g
#SBATCH --ntasks-per-socket=2

unset XDG_RUNTIME_DIR

module unload xalt/2.6.3-gpu
module load singularity/3.2.0
module load PrgEnv/GCC+MPICH/2016-07-22 
module load mpich/3.1.3

mpirun singularity exec --nv /datasets/yjuntao/images/petsc_cuda.simg ./testSolve -fieldsplit_0_ksp_type gmres -fieldsplit_0_pc_type gamg
