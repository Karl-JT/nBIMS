cmake_minimum_required(VERSION 3.10)

project(nse2dmlmcmc_IC_f)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(nse2dmlmcmc_IC_f MLMCMCNSE2D_IC_f.cpp )

set_property(TARGET nse2dmlmcmc_IC_f PROPERTY CXX_STANDARD 14)

target_link_libraries( nse2dmlmcmc_IC_f PUBLIC
            IO
            nse2dsolver
            FEModule
            numericalRecipes
            mcmcLib
			mpi
			petsc
			)

target_include_directories(	nse2dmlmcmc_IC_f PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${PETSC_INCLUDE_DIRS}
				)
