cmake_minimum_required(VERSION 3.10)

project(nse2dcnmlmcmc)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(nse2dcnmlmcmc MLMCMCNSE2DCN.cpp )

set_property(TARGET nse2dcnmlmcmc PROPERTY CXX_STANDARD 14)

target_link_libraries( nse2dcnmlmcmc PUBLIC
            IO
            nse2dsolver
            FEModule
            numericalRecipes
            mcmcLib
			mpi
			petsc
			)

target_include_directories(	nse2dcnmlmcmc PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${PETSC_INCLUDE_DIRS}
				)

