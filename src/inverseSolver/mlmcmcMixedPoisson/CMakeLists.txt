cmake_minimum_required(VERSION 3.10)

project(mixedPoisson2dmlmcmc)

include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(mixedPoisson2dmlmcmc mlmcmcMixedPoisson.cpp )

set_property(TARGET mixedPoisson2dmlmcmc PROPERTY CXX_STANDARD 14)

target_link_libraries( mixedPoisson2dmlmcmc PUBLIC
            IO
            FEModule
            elasticMixed2d
            numericalRecipes
            mcmcLib
			)

target_include_directories(	mixedPoisson2dmlmcmc PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				)
