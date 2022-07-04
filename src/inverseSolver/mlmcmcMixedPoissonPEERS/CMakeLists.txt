cmake_minimum_required(VERSION 3.10)

project(PEERS2dmlmcmc)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(PEERS2dmlmcmc mlmcmcMixedPoissonPEERS.cpp )

set_property(TARGET PEERS2dmlmcmc PROPERTY CXX_STANDARD 14)

target_link_libraries( PEERS2dmlmcmc PUBLIC
            IO
            elasticMixed2dRT
            numericalRecipes
            mcmcLib
	    mpi
	    petsc
			)

target_include_directories(	PEERS2dmlmcmc PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${DOLFIN_INCLUDE_DIRS}
				${PETSC_INCLUDE_DIRS}
				${MPI_CXX_INCLUDE_PATH}$
				)









