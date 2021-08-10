cmake_minimum_required(VERSION 3.10)

project(IO)

file(	GLOB IO_SRC
	"*.h"
	"*.cpp"
	)

add_library(IO ${IO_SRC})

target_link_libraries(IO PUBLIC)

target_include_directories(	IO INTERFACE
				${CMAKE_CURRENT_SOURCE_DIR})
