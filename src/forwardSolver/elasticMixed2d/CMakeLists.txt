cmake_minimum_required(VERSION 3.10)

project(elasticMixed2d)

file(	GLOB elasticMixed2d_SRC
	"*.cpp"
	)

add_library(elasticMixed2d ${elasticMixed2d_SRC} )

set_property(TARGET elasticMixed2d PROPERTY CXX_STANDARD 14)

target_link_libraries( elasticMixed2d PUBLIC
            FEModule
            numericalRecipes
			)

target_include_directories(	elasticMixed2d PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				)


