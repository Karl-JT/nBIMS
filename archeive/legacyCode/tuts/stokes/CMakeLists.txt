cmake_minimum_required(VERSION 3.10)

#set project name
project(stokeTutorial)

find_package(PkgConfig)
pkg_search_module(PETSC REQUIRED PETSc)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

file(	GLOB stokeTutorial_SRC
	"*.h"
	"*cpp"
	)

#add the executable
add_executable(stokeTutorial ${stokeTutorial_SRC}) 

target_link_libraries(  stokeTutorial PUBLIC
			petsc
			mpi
			mcmcLib
			numericalRecipes
			)

target_include_directories(	stokeTutorial INTERFACE 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${PETSC_INCLUDE_DIRS}
				)
