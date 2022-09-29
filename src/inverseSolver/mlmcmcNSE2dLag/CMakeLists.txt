cmake_minimum_required(VERSION 3.10)

project(nse2dLagmlmcmc)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(nse2dLagmlmcmc mlmcmcNSE2dLag.cpp )

set_property(TARGET nse2dLagmlmcmc PROPERTY CXX_STANDARD 14)

target_link_libraries( nse2dLagmlmcmc PUBLIC
            IO
            nse2dsolver
            FEModule
            numericalRecipes
            mcmcLib
	    mpi
	    petsc
	    ${MPI_CXX_LIBRARIES}		)

    target_include_directories(	nse2dLagmlmcmc PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${PETSC_INCLUDE_DIRS}
				${PESTC_CXX_INCLUDE_PATH}
				)
