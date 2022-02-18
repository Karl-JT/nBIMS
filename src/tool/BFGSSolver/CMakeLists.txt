cmake_minimum_required(VERSION 3.10)

project(BFGSSolver)

file(	GLOB BFGSSolver_SRC
	"*.h"
	)

add_library(BFGSSolver INTERFACE)

#target_link_libraries(	BFGSSolver PUBLIC 
#			${BFGSSolver_SRC}
#			Eigen3::Eigen
#			LBFGS
#			)

target_include_directories(	BFGSSolver INTERFACE 
				${CMAKE_CURRENT_SOURCE_DIR}		
				)				
