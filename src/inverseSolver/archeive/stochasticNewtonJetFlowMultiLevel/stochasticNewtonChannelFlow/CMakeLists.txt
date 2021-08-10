cmake_minimum_required(VERSION 3.10)

#set project name
project(inverseSolver)

find_package(MPI REQUIRED)
include_directories(/usr/local/include/adolc)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
link_directories(/usr/local/lib64/)

file(	GLOB inverseSolver_SRC
	"*cpp"
	)

#add the executable
add_executable(inverseSolver ${inverseSolver_SRC}) 

target_link_libraries(	inverseSolver PUBLIC 	
			LBFGS
			BFGSSolver
			bisectionMesh
			Eigen3::Eigen
			${MPI_LIBRARIES}
			oneHierachical
			#			biHierachical
			solverInterface
			mcmcCore
			#			plainMCMC
			#			stochasticNewtonMCMC
			adolc
			)

