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
include src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/depend.make

# Include the progress variables for this target.
include src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/progress.make

# Include the compile flags for this target's objects.
include src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/flags.make

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/flags.make
src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o: src/inverseSolver/stochasticNewtonChannelFlow/stochasticNewtonChannelFlow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o"
	cd /home/fenics/shared/src/inverseSolver/stochasticNewtonChannelFlow && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o -c /home/fenics/shared/src/inverseSolver/stochasticNewtonChannelFlow/stochasticNewtonChannelFlow.cpp

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.i"
	cd /home/fenics/shared/src/inverseSolver/stochasticNewtonChannelFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/shared/src/inverseSolver/stochasticNewtonChannelFlow/stochasticNewtonChannelFlow.cpp > CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.i

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.s"
	cd /home/fenics/shared/src/inverseSolver/stochasticNewtonChannelFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/shared/src/inverseSolver/stochasticNewtonChannelFlow/stochasticNewtonChannelFlow.cpp -o CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.s

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o.requires:

.PHONY : src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o.requires

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o.provides: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o.requires
	$(MAKE) -f src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/build.make src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o.provides.build
.PHONY : src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o.provides

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o.provides.build: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o


# Object files for target stochasticNewtonChannelFlow
stochasticNewtonChannelFlow_OBJECTS = \
"CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o"

# External object files for target stochasticNewtonChannelFlow
stochasticNewtonChannelFlow_EXTERNAL_OBJECTS =

bin/stochasticNewtonChannelFlow: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o
bin/stochasticNewtonChannelFlow: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/build.make
bin/stochasticNewtonChannelFlow: src/forwardSolver/bisectionMesh/libbisectionMesh.a
bin/stochasticNewtonChannelFlow: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
bin/stochasticNewtonChannelFlow: /usr/lib/x86_64-linux-gnu/libmpich.so
bin/stochasticNewtonChannelFlow: src/mcmcVersions/mcmcCore/libmcmcCore.a
bin/stochasticNewtonChannelFlow: src/mcmcVersions/stochasticNewtonMCMC/libstochasticNewtonMCMC.a
bin/stochasticNewtonChannelFlow: src/mcmcVersions/mcmcCore/libmcmcCore.a
bin/stochasticNewtonChannelFlow: src/forwardSolver/bisectionMesh/libbisectionMesh.a
bin/stochasticNewtonChannelFlow: src/forwardSolver/dnsDataInterpreter/libdnsDataInterpreter.a
bin/stochasticNewtonChannelFlow: src/tool/IO/libIO.a
bin/stochasticNewtonChannelFlow: src/forwardSolver/jetFlow/libjetFlow.a
bin/stochasticNewtonChannelFlow: /usr/local/lib/libdolfin.so.2019.1.0
bin/stochasticNewtonChannelFlow: /usr/lib/x86_64-linux-gnu/libboost_timer.so
bin/stochasticNewtonChannelFlow: /usr/lib/x86_64-linux-gnu/hdf5/mpich/libhdf5.so
bin/stochasticNewtonChannelFlow: /usr/lib/x86_64-linux-gnu/libsz.so
bin/stochasticNewtonChannelFlow: /usr/lib/x86_64-linux-gnu/libz.so
bin/stochasticNewtonChannelFlow: /usr/lib/x86_64-linux-gnu/libdl.so
bin/stochasticNewtonChannelFlow: /usr/lib/x86_64-linux-gnu/libm.so
bin/stochasticNewtonChannelFlow: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
bin/stochasticNewtonChannelFlow: /usr/lib/x86_64-linux-gnu/libmpich.so
bin/stochasticNewtonChannelFlow: /usr/local/slepc-32/lib/libslepc.so
bin/stochasticNewtonChannelFlow: /usr/local/petsc-32/lib/libpetsc.so
bin/stochasticNewtonChannelFlow: src/tool/linearAlgebra/liblinearAlgebra.a
bin/stochasticNewtonChannelFlow: /usr/lib/x86_64-linux-gnu/libgsl.so
bin/stochasticNewtonChannelFlow: /usr/lib/x86_64-linux-gnu/libgslcblas.so
bin/stochasticNewtonChannelFlow: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/stochasticNewtonChannelFlow"
	cd /home/fenics/shared/src/inverseSolver/stochasticNewtonChannelFlow && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stochasticNewtonChannelFlow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/build: bin/stochasticNewtonChannelFlow

.PHONY : src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/build

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/requires: src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/stochasticNewtonChannelFlow.cpp.o.requires

.PHONY : src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/requires

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/clean:
	cd /home/fenics/shared/src/inverseSolver/stochasticNewtonChannelFlow && $(CMAKE_COMMAND) -P CMakeFiles/stochasticNewtonChannelFlow.dir/cmake_clean.cmake
.PHONY : src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/clean

src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/depend:
	cd /home/fenics/shared && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fenics/shared /home/fenics/shared/src/inverseSolver/stochasticNewtonChannelFlow /home/fenics/shared /home/fenics/shared/src/inverseSolver/stochasticNewtonChannelFlow /home/fenics/shared/src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/inverseSolver/stochasticNewtonChannelFlow/CMakeFiles/stochasticNewtonChannelFlow.dir/depend

