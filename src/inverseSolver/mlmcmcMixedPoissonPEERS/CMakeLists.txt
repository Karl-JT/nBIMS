cmake_minimum_required(VERSION 3.10)

project(PEERS2dmlmcmc)

add_executable(PEERS2dmlmcmc mlmcmcMixedPoissonPEERS.cpp )

set_property(TARGET PEERS2dmlmcmc PROPERTY CXX_STANDARD 14)

target_link_libraries( PEERS2dmlmcmc PUBLIC
            IO
            elasticMixed2dRT
            numericalRecipes
            mcmcLib
			)

target_include_directories(	PEERS2dmlmcmc PUBLIC 
				${CMAKE_CURRENT_SOURCE_DIR} 
				${DOLFIN_INCLUDE_DIRS}
				)
