cmake_minimum_required(VERSION 3.10)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

project(stokes2dsolver)

file(	GLOB stokes2dsolver_SRC
	"*.cpp"
	)

add_library(stokes2dsolver ${stokes2dsolver_SRC} )

set_property(TARGET stokes2dsolver PROPERTY CXX_STANDARD 14)

target_link_libraries( stokes2dsolver PUBLIC
            FEModule
            numericalRecipes
			petsc
			mpi
			)

target_include_directories(	stokes2dsolver PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${PETSC_INCLUDE_DIRS}
				)


