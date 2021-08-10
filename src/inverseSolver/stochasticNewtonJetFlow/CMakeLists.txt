cmake_minimum_required(VERSION 3.10)

#set project name
project(stochasticNewtonJetFlow)

find_package(MPI REQUIRED)
include_directories(/usr/local/include/adolc)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
link_directories(/usr/local/lib64/)

file(	GLOB inverseSolver_SRC
	"*.h"
	"*cpp"
	)

#add the executable
add_executable(stochasticNewtonJetFlow ${inverseSolver_SRC}) 

target_link_libraries(	stochasticNewtonJetFlow PUBLIC 	
			bisectionMesh
			${MPI_LIBRARIES}
			mcmcCore
			stochasticNewtonMCMC
			)

