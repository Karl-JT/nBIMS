cmake_minimum_required(VERSION 3.10)

project(mcmcLib)

add_library(mcmcLib INTERFACE)
target_include_directories(mcmcLib INTERFACE
				${CMAKE_CURRENT_SOURCE_DIR}/include
				)
