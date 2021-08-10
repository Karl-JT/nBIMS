cmake_minimum_required(VERSION 3.10)

#set project name
project(nseTutorial)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

file(	GLOB nseTutorial_SRC
	"*.h"
	"*cpp"
	)

#add the executable
add_executable(nseTutorial ${nseTutorial_SRC}) 

target_link_libraries(  nseTutorial PUBLIC
            IO
			petsc
			mpi
			mcmcLib
			numericalRecipes
			)

target_include_directories(	nseTutorial INTERFACE 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${PETSC_INCLUDE_DIRS}
				)
