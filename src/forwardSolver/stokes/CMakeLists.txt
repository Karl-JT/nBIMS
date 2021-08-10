cmake_minimum_required(VERSION 3.10)

project(stokeSolver)

find_package(PkgConfig)
pkg_search_module(PETSC REQUIRED PETSc)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(testSolve stokes_v1-1.cpp )

set_property(TARGET testSolve PROPERTY CXX_STANDARD 14)

target_link_libraries(  testSolve PUBLIC
			petsc
			mpi
			)

target_include_directories(	testSolve INTERFACE 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${PETSC_INCLUDE_DIRS}
				)


