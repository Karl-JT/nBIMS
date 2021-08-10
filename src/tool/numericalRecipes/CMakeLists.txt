cmake_minimum_required(VERSION 3.10)

project(numericalRecipes)

file(	GLOB numericalRecipes_SRC
	"*.h"
	"*.cpp"
	)

add_library(numericalRecipes ${numericalRecipes_SRC})

target_link_libraries(	numericalRecipes PUBLIC 
			${GSL_LIBRARIES}
			)

target_include_directories(	numericalRecipes INTERFACE
				${CMAKE_CURRENT_SOURCE_DIR}
				${GSL_INCLUDE_DIRS})
