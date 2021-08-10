cmake_minimum_required(VERSION 3.10)

#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}:/usr/include/gsl)
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}:/usr/lib/x86_64-linux-gnu)

project(oneHierachical)

find_package(GSL)

file(	GLOB oneHierachical_INTERFACE
	"*.h"
	)

add_library(oneHierachical INTERFACE)

target_link_libraries(	oneHierachical INTERFACE
			mcmcCore
			)

target_include_directories(	oneHierachical INTERFACE
				${CMAKE_CURRENT_SOURCE_DIR}
				)
