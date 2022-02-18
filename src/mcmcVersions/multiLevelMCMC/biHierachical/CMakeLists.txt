cmake_minimum_required(VERSION 3.10)

#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}:/usr/include/gsl)
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}:/usr/lib/x86_64-linux-gnu)

project(biHierachical)

find_package(GSL)

file(	GLOB biHierachical_SRC
	"*.h"
	"*.cpp"
	)

add_library(biHierachical ${biHierachical_SRC})

target_link_libraries(	biHierachical PUBLIC
			mcmcCore
			)

target_include_directories(	biHierachical PUBLIC
				${CMAKE_CURRENT_SOURCE_DIR}
				)
