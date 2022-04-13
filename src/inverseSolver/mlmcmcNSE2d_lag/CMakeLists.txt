cmake_minimum_required(VERSION 3.10)

project(nse2dmlmcmc)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(nse2dmlmcmc MLMCMCNSE2D.cpp )

set_property(TARGET nse2dmlmcmc PROPERTY CXX_STANDARD 14)

target_link_libraries( nse2dmlmcmc PUBLIC
            IO
            nse2dsolver
            FEModule
            numericalRecipes
            mcmcLib
			mpi
			petsc
			)

target_include_directories(	nse2dmlmcmc PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${PETSC_INCLUDE_DIRS}
				)
