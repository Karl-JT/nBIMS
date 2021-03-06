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
CMAKE_SOURCE_DIR = /home/fenics/share

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fenics/share

# Include any dependencies generated for this target.
include src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/depend.make

# Include the progress variables for this target.
include src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/progress.make

# Include the compile flags for this target's objects.
include src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/flags.make

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/flags.make
src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o: src/inverseSolver/stochasticNewtonChannelFlow/inverseSolver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/share/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o"
	cd /home/fenics/share/src/inverseSolver/stochasticNewtonChannelFlow && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o -c /home/fenics/share/src/inverseSolver/stochasticNewtonChannelFlow/inverseSolver.cpp

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inverseSolver.dir/inverseSolver.cpp.i"
	cd /home/fenics/share/src/inverseSolver/stochasticNewtonChannelFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/share/src/inverseSolver/stochasticNewtonChannelFlow/inverseSolver.cpp > CMakeFiles/inverseSolver.dir/inverseSolver.cpp.i

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inverseSolver.dir/inverseSolver.cpp.s"
	cd /home/fenics/share/src/inverseSolver/stochasticNewtonChannelFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/share/src/inverseSolver/stochasticNewtonChannelFlow/inverseSolver.cpp -o CMakeFiles/inverseSolver.dir/inverseSolver.cpp.s

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o.requires:

.PHONY : src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o.requires

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o.provides: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o.requires
	$(MAKE) -f src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/build.make src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o.provides.build
.PHONY : src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o.provides

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o.provides.build: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o


# Object files for target inverseSolver
inverseSolver_OBJECTS = \
"CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o"

# External object files for target inverseSolver
inverseSolver_EXTERNAL_OBJECTS =

bin/inverseSolver: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o
bin/inverseSolver: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/build.make
bin/inverseSolver: src/forwardSolver/bisectionMesh/libbisectionMesh.a
bin/inverseSolver: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
bin/inverseSolver: /usr/lib/x86_64-linux-gnu/libmpich.so
bin/inverseSolver: src/mcmcVersions/mcmcCore/libmcmcCore.a
bin/inverseSolver: src/forwardSolver/bisectionMesh/libbisectionMesh.a
bin/inverseSolver: src/forwardSolver/dnsDataInterpreter/libdnsDataInterpreter.a
bin/inverseSolver: src/forwardSolver/jetFlow/libjetFlow.a
bin/inverseSolver: /usr/local/lib/libdolfin.so.2019.1.0
bin/inverseSolver: /usr/lib/x86_64-linux-gnu/libboost_timer.so
bin/inverseSolver: /usr/lib/x86_64-linux-gnu/hdf5/mpich/libhdf5.so
bin/inverseSolver: /usr/lib/x86_64-linux-gnu/libsz.so
bin/inverseSolver: /usr/lib/x86_64-linux-gnu/libz.so
bin/inverseSolver: /usr/lib/x86_64-linux-gnu/libdl.so
bin/inverseSolver: /usr/lib/x86_64-linux-gnu/libm.so
bin/inverseSolver: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
bin/inverseSolver: /usr/lib/x86_64-linux-gnu/libmpich.so
bin/inverseSolver: /usr/local/slepc-32/lib/libslepc.so
bin/inverseSolver: /usr/local/petsc-32/lib/libpetsc.so
bin/inverseSolver: src/tool/linearAlgebra/liblinearAlgebra.a
bin/inverseSolver: /usr/lib/x86_64-linux-gnu/libgsl.so
bin/inverseSolver: /usr/lib/x86_64-linux-gnu/libgslcblas.so
bin/inverseSolver: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fenics/share/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/inverseSolver"
	cd /home/fenics/share/src/inverseSolver/stochasticNewtonChannelFlow && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/inverseSolver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/build: bin/inverseSolver

.PHONY : src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/build

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/requires: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/inverseSolver.cpp.o.requires

.PHONY : src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/requires

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/clean:
	cd /home/fenics/share/src/inverseSolver/stochasticNewtonChannelFlow && $(CMAKE_COMMAND) -P CMakeFiles/inverseSolver.dir/cmake_clean.cmake
.PHONY : src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/clean

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/depend:
	cd /home/fenics/share && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fenics/share /home/fenics/share/src/inverseSolver/stochasticNewtonChannelFlow /home/fenics/share /home/fenics/share/src/inverseSolver/stochasticNewtonChannelFlow /home/fenics/share/src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/inverseSolver.dir/depend

