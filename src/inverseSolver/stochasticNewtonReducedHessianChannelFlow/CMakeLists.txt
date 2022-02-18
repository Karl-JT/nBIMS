cmake_minimum_required(VERSION 3.10)

#set project name
project(stochasticNewtonReducedHessianChannelFlow)

find_package(MPI REQUIRED)
include_directories(/usr/local/include/adolc)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
link_directories(/usr/local/lib64/)

file(	GLOB stochasticNewtonReducedHessianChannelFlow_SRC
	"*.h"
	"*cpp"
	)

#add the executable
add_executable(stochasticNewtonReducedHessianChannelFlow ${stochasticNewtonReducedHessianChannelFlow_SRC}) 

target_link_libraries(	stochasticNewtonReducedHessianChannelFlow PUBLIC 	
			reducedSolver
			${MPI_LIBRARIES}
			mcmcCore
			stochasticNewtonMCMC
			)

