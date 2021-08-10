cmake_minimum_required(VERSION 3.10)

#set project name
project(forwardCavityFlow)

find_package(MPI REQUIRED)
include_directories(/usr/local/include/adolc)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
link_directories(/usr/local/lib64/)

file(	GLOB forwardCavity_SRC
	"*cpp"
	)

#add the executable
add_executable(forwardCavityFlow ${forwardCavity_SRC}) 

target_link_libraries(	forwardCavityFlow PUBLIC 	
			cavityFlow
			Eigen3::Eigen
			${MPI_LIBRARIES}
			oneHierachical
			mcmcCore
			adolc
			)

