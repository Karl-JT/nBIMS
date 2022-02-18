cmake_minimum_required(VERSION 3.10)

project(dnsDataInterpreter)

file (	GLOB dnsDataInterpreter_SRC
	"*.h"
	"*.cpp"
	)

add_library(dnsDataInterpreter ${dnsDataInterpreter_SRC})

target_include_directories(dnsDataInterpreter INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

