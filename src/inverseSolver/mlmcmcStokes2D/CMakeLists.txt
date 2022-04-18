cmake_minimum_required(VERSION 3.10)

project(stokes2dmlmcmc)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(stokes2dmlmcmc mlmcmcStokes2D.cpp )

set_property(TARGET stokes2dmlmcmc PROPERTY CXX_STANDARD 14)

target_link_libraries( stokes2dmlmcmc PUBLIC
            IO
            FEModule
            stokes2dsolver
            numericalRecipes
            mcmcLib
	    mpi
	    petsc
	    ${MPI_CXX_LIBRARIES}
			)

target_include_directories(	stokes2dmlmcmc PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${PETSC_INCLUDE_DIRS}
				${MPI_CXX_INCLUDE_PATH}
				)
