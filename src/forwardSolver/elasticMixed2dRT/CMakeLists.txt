cmake_minimum_required(VERSION 3.10)

project(elasticMixed2dRT)

file(	GLOB elasticMixed2dRT_SRC
	"*.cpp"
	)

add_library(elasticMixed2dRT ${elasticMixed2dRT_SRC} )

set_property(TARGET elasticMixed2dRT PROPERTY CXX_STANDARD 14)

target_link_libraries( elasticMixed2dRT PUBLIC
            numericalRecipes
	    dolfin
			)

target_include_directories(	elasticMixed2dRT PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${DOLFIN_INCLUDE_DIRS}
				)
