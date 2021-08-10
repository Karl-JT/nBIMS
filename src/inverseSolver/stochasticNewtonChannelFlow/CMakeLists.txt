cmake_minimum_required(VERSION 3.10)

#set project name
project(stochasticNewtonChannelFlow)

find_package(MPI REQUIRED)
include_directories(/usr/local/include/adolc)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
link_directories(/usr/local/lib64/)

file(	GLOB stochasticNewtonChannel_SRC
	"*.h"
	"*cpp"
	)

#add the executable
add_executable(stochasticNewtonChannelFlow ${stochasticNewtonChannel_SRC}) 

target_link_libraries(	stochasticNewtonChannelFlow PUBLIC 	
			bisectionMesh
			${MPI_LIBRARIES}
			mcmcCore
			stochasticNewtonMCMC
			)

