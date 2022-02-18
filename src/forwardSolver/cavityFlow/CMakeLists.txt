cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME cavityFlow)
project(${PROJECT_NAME})

find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# Do not throw error for 'multi-line comments' (these are typical in
# rst which includes LaTeX)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment" HAVE_NO_MULTLINE)
if (HAVE_NO_MULTLINE)
  set(CMAKE_CXX_FLAGS "-Wno-comment ${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "-Wno-unused-parameter ${CMAKE_CXX_FLAGS}")

file(GLOB cavityFlow_SRC
		"*.h"
		"*.cpp"
)

add_library(${PROJECT_NAME} ${cavityFlow_SRC})

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfin
		      IO	)

target_include_directories(		${PROJECT_NAME} INTERFACE
								${CMAKE_CURRENT_SOURCE_DIR}
								IO					)

