cmake_minimum_required(VERSION 3.10)

project(nse2dPetscFEmlmcmc)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(nse2dPetscFEmlmcmc mlmcmcNSE2dPetscFE.cpp )

set_property(TARGET nse2dPetscFEmlmcmc PROPERTY CXX_STANDARD 14)

target_link_libraries( nse2dPetscFEmlmcmc PUBLIC
            IO
            nse2dsolver
            FEModule
            numericalRecipes
            mcmcLib
	    mpi
	    petsc
            ${MPI_CXX_LIBRARIES}
			)

target_include_directories(	nse2dPetscFEmlmcmc PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${PETSC_INCLUDE_DIRS}
                		${MPI_CXX_INCLUDE_PATH}
				)
