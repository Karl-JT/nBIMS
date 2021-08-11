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
include src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/depend.make

# Include the progress variables for this target.
include src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/progress.make

# Include the compile flags for this target's objects.
include src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/flags.make

src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o: src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/flags.make
src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o: src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/inverseSolver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o"
	cd /home/fenics/shared/src/inverseSolver/stochasticNewtonReducedHessianChannelFlow && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o -c /home/fenics/shared/src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/inverseSolver.cpp

src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.i"
	cd /home/fenics/shared/src/inverseSolver/stochasticNewtonReducedHessianChannelFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/shared/src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/inverseSolver.cpp > CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.i

src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.s"
	cd /home/fenics/shared/src/inverseSolver/stochasticNewtonReducedHessianChannelFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/shared/src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/inverseSolver.cpp -o CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.s

src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o.requires:

.PHONY : src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o.requires

src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o.provides: src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o.requires
	$(MAKE) -f src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/build.make src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o.provides.build
.PHONY : src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o.provides

src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o.provides.build: src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o


# Object files for target stochasticNewtonReducedHessianChannelFlow
stochasticNewtonReducedHessianChannelFlow_OBJECTS = \
"CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o"

# External object files for target stochasticNewtonReducedHessianChannelFlow
stochasticNewtonReducedHessianChannelFlow_EXTERNAL_OBJECTS =

bin/stochasticNewtonReducedHessianChannelFlow: src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o
bin/stochasticNewtonReducedHessianChannelFlow: src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/build.make
bin/stochasticNewtonReducedHessianChannelFlow: src/forwardSolver/channelFlow/libreducedSolver.a
bin/stochasticNewtonReducedHessianChannelFlow: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
bin/stochasticNewtonReducedHessianChannelFlow: /usr/lib/x86_64-linux-gnu/libmpich.so
bin/stochasticNewtonReducedHessianChannelFlow: src/mcmcVersions/mcmcCore/libmcmcCore.a
bin/stochasticNewtonReducedHessianChannelFlow: src/mcmcVersions/stochasticNewtonMCMC/libstochasticNewtonMCMC.a
bin/stochasticNewtonReducedHessianChannelFlow: src/mcmcVersions/mcmcCore/libmcmcCore.a
bin/stochasticNewtonReducedHessianChannelFlow: src/forwardSolver/bisectionMesh/libbisectionMesh.a
bin/stochasticNewtonReducedHessianChannelFlow: src/forwardSolver/dnsDataInterpreter/libdnsDataInterpreter.a
bin/stochasticNewtonReducedHessianChannelFlow: src/forwardSolver/jetFlow/libjetFlow.a
bin/stochasticNewtonReducedHessianChannelFlow: /usr/local/lib/libdolfin.so.2019.1.0
bin/stochasticNewtonReducedHessianChannelFlow: /usr/lib/x86_64-linux-gnu/libboost_timer.so
bin/stochasticNewtonReducedHessianChannelFlow: /usr/lib/x86_64-linux-gnu/hdf5/mpich/libhdf5.so
bin/stochasticNewtonReducedHessianChannelFlow: /usr/lib/x86_64-linux-gnu/libsz.so
bin/stochasticNewtonReducedHessianChannelFlow: /usr/lib/x86_64-linux-gnu/libz.so
bin/stochasticNewtonReducedHessianChannelFlow: /usr/lib/x86_64-linux-gnu/libdl.so
bin/stochasticNewtonReducedHessianChannelFlow: /usr/lib/x86_64-linux-gnu/libm.so
bin/stochasticNewtonReducedHessianChannelFlow: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
bin/stochasticNewtonReducedHessianChannelFlow: /usr/lib/x86_64-linux-gnu/libmpich.so
bin/stochasticNewtonReducedHessianChannelFlow: /usr/local/slepc-32/lib/libslepc.so
bin/stochasticNewtonReducedHessianChannelFlow: /usr/local/petsc-32/lib/libpetsc.so
bin/stochasticNewtonReducedHessianChannelFlow: src/tool/linearAlgebra/liblinearAlgebra.a
bin/stochasticNewtonReducedHessianChannelFlow: /usr/lib/x86_64-linux-gnu/libgsl.so
bin/stochasticNewtonReducedHessianChannelFlow: /usr/lib/x86_64-linux-gnu/libgslcblas.so
bin/stochasticNewtonReducedHessianChannelFlow: src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/stochasticNewtonReducedHessianChannelFlow"
	cd /home/fenics/shared/src/inverseSolver/stochasticNewtonReducedHessianChannelFlow && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/build: bin/stochasticNewtonReducedHessianChannelFlow

.PHONY : src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/build

src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/requires: src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/inverseSolver.cpp.o.requires

.PHONY : src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/requires

src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/clean:
	cd /home/fenics/shared/src/inverseSolver/stochasticNewtonReducedHessianChannelFlow && $(CMAKE_COMMAND) -P CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/cmake_clean.cmake
.PHONY : src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/clean

src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/depend:
	cd /home/fenics/shared && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fenics/shared /home/fenics/shared/src/inverseSolver/stochasticNewtonReducedHessianChannelFlow /home/fenics/shared /home/fenics/shared/src/inverseSolver/stochasticNewtonReducedHessianChannelFlow /home/fenics/shared/src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/inverseSolver/stochasticNewtonReducedHessianChannelFlow/CMakeFiles/stochasticNewtonReducedHessianChannelFlow.dir/depend
