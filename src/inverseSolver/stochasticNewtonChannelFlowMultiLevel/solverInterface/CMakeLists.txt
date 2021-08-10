cmake_minimum_required(VERSION 3.10)

#set project name
project(solverInterface)

#add the executable
add_library(solverInterface INTERFACE) 
	
target_include_directories(	solverInterface INTERFACE
				${CMAKE_CURRENT_SOURCE_DIR}
				jetSolver
				#bisectionMesh
				)

