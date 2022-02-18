cmake_minimum_required(VERSION 3.10)

project(plainMCMC)

file(GLOB plainMCMC_SRC
	"*.h"
	"*.cpp"
)

add_library(plainMCMC ${plainMCMC_SRC})

target_link_libraries(	plainMCMC PUBLIC 
			mcmcCore
			${GSL_LIBRARIES}
			solverInterface
			)

target_include_directories(	plainMCMC PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR}
				mcmcCore
				${GSL_INCLUDE_DIRS}
				solverInterface
				)


