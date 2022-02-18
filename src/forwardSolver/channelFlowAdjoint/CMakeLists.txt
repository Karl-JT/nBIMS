cmake_minimum_required(VERSION 3.10)

project(channelFlowAdjoint)

find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

set(CMAKE_CXX_FLAGS "-Wno-maybe-uninitialized ${CMAKE_CXX_FLAGS}")

file(GLOB channelFlowAdjoint_SRC
	"*.h"
	"*.cpp"
)

add_library(channelFlowAdjoint ${channelFlowAdjoint_SRC})
set_property(TARGET channelFlowAdjoint PROPERTY CXX_STANDARD 14)

target_link_libraries(	channelFlowAdjoint PUBLIC 
			dolfin
			dnsDataInterpreter
			adolc
			)

target_include_directories(	channelFlowAdjoint INTERFACE 
				${CMAKE_CURRENT_SOURCE_DIR} 
				dnsDataInterpreter
				adolc
				)
