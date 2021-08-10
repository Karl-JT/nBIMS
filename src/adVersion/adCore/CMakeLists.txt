cmake_minimum_required(VERSION 3.10)

project(adCore)

file(	GLOB adCore_SRC
	"*.h"
	"*.cpp"
	)

add_library(adCore ${adCore_SRC})

target_link_libraries(	adCore PUBLIC 
			bisectionMesh
			)

target_include_directories(	adCore INTERFACE
				${CMAKE_CURRENT_SOURCE_DIR}
				)
