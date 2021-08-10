cmake_minimum_required(VERSION 3.10)

#set project name
project(stochasticNewtonJetFlowML)

find_package(MPI REQUIRED)
include_directories(/usr/local/include/adolc)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
link_directories(/usr/local/lib64/)

file(	GLOB inverseSolver_SRC
	"*cpp"
	)

#add the executable
add_executable(stochasticNewtonJetFlowML ${inverseSolver_SRC}) 

target_link_libraries(	stochasticNewtonJetFlowML PUBLIC 	
			Eigen3::Eigen
			${MPI_LIBRARIES}
			oneHierachical
			mcmcCore
			adolc
			)

