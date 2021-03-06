# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/fenics/shared

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fenics/shared

# Include any dependencies generated for this target.
include src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/depend.make

# Include the progress variables for this target.
include src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/progress.make

# Include the compile flags for this target's objects.
include src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/flags.make

src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o: src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/flags.make
src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o: src/inverseSolver/forwardJetFlow/forwardJetFlow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o"
	cd /home/fenics/shared/src/inverseSolver/forwardJetFlow && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o -c /home/fenics/shared/src/inverseSolver/forwardJetFlow/forwardJetFlow.cpp

src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.i"
	cd /home/fenics/shared/src/inverseSolver/forwardJetFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/shared/src/inverseSolver/forwardJetFlow/forwardJetFlow.cpp > CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.i

src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.s"
	cd /home/fenics/shared/src/inverseSolver/forwardJetFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/shared/src/inverseSolver/forwardJetFlow/forwardJetFlow.cpp -o CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.s

src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o.requires:

.PHONY : src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o.requires

src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o.provides: src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o.requires
	$(MAKE) -f src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/build.make src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o.provides.build
.PHONY : src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o.provides

src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o.provides.build: src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o


# Object files for target forwardJetFlow
forwardJetFlow_OBJECTS = \
"CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o"

# External object files for target forwardJetFlow
forwardJetFlow_EXTERNAL_OBJECTS =

bin/forwardJetFlow: src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o
bin/forwardJetFlow: src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/build.make
bin/forwardJetFlow: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
bin/forwardJetFlow: /usr/lib/x86_64-linux-gnu/libmpich.so
bin/forwardJetFlow: src/mcmcVersions/mcmcCore/libmcmcCore.a
bin/forwardJetFlow: src/forwardSolver/bisectionMesh/libbisectionMesh.a
bin/forwardJetFlow: src/forwardSolver/dnsDataInterpreter/libdnsDataInterpreter.a
bin/forwardJetFlow: src/forwardSolver/jetFlow/libjetFlow.a
bin/forwardJetFlow: src/tool/IO/libIO.a
bin/forwardJetFlow: /usr/local/lib/libdolfin.so.2019.1.0
bin/forwardJetFlow: /usr/lib/x86_64-linux-gnu/libboost_timer.so
bin/forwardJetFlow: /usr/lib/x86_64-linux-gnu/hdf5/mpich/libhdf5.so
bin/forwardJetFlow: /usr/lib/x86_64-linux-gnu/libsz.so
bin/forwardJetFlow: /usr/lib/x86_64-linux-gnu/libz.so
bin/forwardJetFlow: /usr/lib/x86_64-linux-gnu/libdl.so
bin/forwardJetFlow: /usr/lib/x86_64-linux-gnu/libm.so
bin/forwardJetFlow: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
bin/forwardJetFlow: /usr/lib/x86_64-linux-gnu/libmpich.so
bin/forwardJetFlow: /usr/local/slepc-32/lib/libslepc.so
bin/forwardJetFlow: /usr/local/petsc-32/lib/libpetsc.so
bin/forwardJetFlow: src/tool/linearAlgebra/liblinearAlgebra.a
bin/forwardJetFlow: /usr/lib/x86_64-linux-gnu/libgsl.so
bin/forwardJetFlow: /usr/lib/x86_64-linux-gnu/libgslcblas.so
bin/forwardJetFlow: src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/forwardJetFlow"
	cd /home/fenics/shared/src/inverseSolver/forwardJetFlow && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/forwardJetFlow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/build: bin/forwardJetFlow

.PHONY : src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/build

src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/requires: src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/forwardJetFlow.cpp.o.requires

.PHONY : src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/requires

src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/clean:
	cd /home/fenics/shared/src/inverseSolver/forwardJetFlow && $(CMAKE_COMMAND) -P CMakeFiles/forwardJetFlow.dir/cmake_clean.cmake
.PHONY : src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/clean

src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/depend:
	cd /home/fenics/shared && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fenics/shared /home/fenics/shared/src/inverseSolver/forwardJetFlow /home/fenics/shared /home/fenics/shared/src/inverseSolver/forwardJetFlow /home/fenics/shared/src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/inverseSolver/forwardJetFlow/CMakeFiles/forwardJetFlow.dir/depend

