cmake_minimum_required(VERSION 3.10)

find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} /usr/local/petsc-32/lib/pkgconfig/:/usr/local/lib/pkgconfig/:$ENV{PKG_CONFIG_PATH})
pkg_search_module(PETSC REQUIRED PETSc)

project(FEModule)

file(	GLOB FEModule_SRC
	"*.cpp"
	)

add_library(FEModule ${FEModule_SRC})

target_link_libraries(FEModule PUBLIC
                        mpi
                        petsc
                        numericalRecipes)

target_include_directories(	FEModule PUBLIC
				${CMAKE_CURRENT_SOURCE_DIR}
                ${MPI_INCLUDE_PATH}
                ${PETSC_INCLUDE_DIRS})