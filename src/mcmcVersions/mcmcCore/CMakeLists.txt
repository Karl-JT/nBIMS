cmake_minimum_required(VERSION 3.10)

project(mcmcCore)

file(	GLOB mcmcCore_SRC
	"*.h"
	"*.cpp"
	)

add_library(mcmcCore ${mcmcCore_SRC})

target_link_libraries(	mcmcCore PUBLIC
			bisectionMesh
			jetFlow
			${GSL_LIBRARIES}
			linearAlgebra
			)

target_include_directories(	mcmcCore PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR}
				bisectionMesh
				jetFlow
				linearAlgebra
				${GSL_INCLUDE_DIRS}
				)

