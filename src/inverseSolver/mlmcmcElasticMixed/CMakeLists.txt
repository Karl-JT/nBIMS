cmake_minimum_required(VERSION 3.10)

project(elasticMixed2dmlmcmc)

include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(elasticMixed2dmlmcmc mlmcmcElasticMixed.cpp )

set_property(TARGET elasticMixed2dmlmcmc PROPERTY CXX_STANDARD 14)

target_link_libraries( elasticMixed2dmlmcmc PUBLIC
            IO
            FEModule
            elasticMixed2d
            numericalRecipes
            mcmcLib
	    ${MPI_CXX_LIBRARIES}
			)

target_include_directories(	elasticMixed2dmlmcmc PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${MPI_CXX_INCLUDE_PATH}
				)
