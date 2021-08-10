cmake_minimum_required(VERSION 3.10)

project(bisectionMesh)

file(GLOB bisectionMesh_SRC
	"*.h"
	"*.cpp"
)

add_library(bisectionMesh ${bisectionMesh_SRC})

target_link_libraries(	bisectionMesh PUBLIC 
			dnsDataInterpreter
			adolc
			IO
			)

target_include_directories(	bisectionMesh INTERFACE 
				${CMAKE_CURRENT_SOURCE_DIR} 
				dnsDataInterpreter
				adolc
				IO
				)
