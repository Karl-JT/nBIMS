cmake_minimum_required(VERSION 3.10)

project(stokes2dpenaltymlmcmc)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(stokes2dpenaltymlmcmc mlmcmcStokes2Dpenalty.cpp )

set_property(TARGET stokes2dpenaltymlmcmc PROPERTY CXX_STANDARD 14)

target_link_libraries( stokes2dpenaltymlmcmc PUBLIC
            IO
            FEModule
            stokes2dsolver
            numericalRecipes
            mcmcLib
	    mpi
            petsc
	    ${MPI_CXX_LIBRARIES})

target_include_directories(	stokes2dpenaltymlmcmc PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${PETSC_INCLUDE_DIRS}
				${MPI_CXX_INCLUDE_PATH}
				)
