cmake_minimum_required(VERSION 3.10)

#set project name
project(forwardJetFlow)

find_package(MPI REQUIRED)
include_directories(/usr/local/include/adolc)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
link_directories(/usr/local/lib64/)

file(	GLOB forwardJetFlow_SRC
	"*cpp"
	)

#add the executable
add_executable(forwardJetFlow ${forwardJetFlow_SRC}) 

target_link_libraries(	forwardJetFlow PUBLIC 	
			Eigen3::Eigen
			${MPI_LIBRARIES}
			oneHierachical
			mcmcCore
			adolc
			)

