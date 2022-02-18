cmake_minimum_required(VERSION 3.10)

project(reducedSolver)

file(GLOB reducedSolver_SRC
	"*.h"
	"*.cpp"
)

add_library(reducedSolver ${reducedSolver_SRC})

target_link_libraries(	reducedSolver PUBLIC 
			dnsDataInterpreter
			adolc
			)

target_include_directories(	reducedSolver INTERFACE 
				${CMAKE_CURRENT_SOURCE_DIR} 
				dnsDataInterpreter
				adolc
				)
