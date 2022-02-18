cmake_minimum_required(VERSION 3.10)

project(stochasticNewtonMCMC)

file(GLOB stochasticNewtonMCMC_SRC
	"*.h"
	"*.cpp"
)

add_library(stochasticNewtonMCMC ${stochasticNewtonMCMC_SRC})

target_link_libraries(	stochasticNewtonMCMC PUBLIC 
			mcmcCore
			)

target_include_directories(	stochasticNewtonMCMC PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				mcmcCore
				)


