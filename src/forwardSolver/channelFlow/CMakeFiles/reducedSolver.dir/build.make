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
include src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/depend.make

# Include the progress variables for this target.
include src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/progress.make

# Include the compile flags for this target's objects.
include src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/flags.make

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o: src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/flags.make
src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o: src/forwardSolver/channelFlow/ChannelCppSolverBiMesh.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o"
	cd /home/fenics/shared/src/forwardSolver/channelFlow && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o -c /home/fenics/shared/src/forwardSolver/channelFlow/ChannelCppSolverBiMesh.cpp

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.i"
	cd /home/fenics/shared/src/forwardSolver/channelFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/shared/src/forwardSolver/channelFlow/ChannelCppSolverBiMesh.cpp > CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.i

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.s"
	cd /home/fenics/shared/src/forwardSolver/channelFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/shared/src/forwardSolver/channelFlow/ChannelCppSolverBiMesh.cpp -o CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.s

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o.requires:

.PHONY : src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o.requires

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o.provides: src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o.requires
	$(MAKE) -f src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/build.make src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o.provides.build
.PHONY : src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o.provides

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o.provides.build: src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o


src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o: src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/flags.make
src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o: src/forwardSolver/channelFlow/turbSolver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o"
	cd /home/fenics/shared/src/forwardSolver/channelFlow && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/reducedSolver.dir/turbSolver.cpp.o -c /home/fenics/shared/src/forwardSolver/channelFlow/turbSolver.cpp

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reducedSolver.dir/turbSolver.cpp.i"
	cd /home/fenics/shared/src/forwardSolver/channelFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/shared/src/forwardSolver/channelFlow/turbSolver.cpp > CMakeFiles/reducedSolver.dir/turbSolver.cpp.i

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reducedSolver.dir/turbSolver.cpp.s"
	cd /home/fenics/shared/src/forwardSolver/channelFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/shared/src/forwardSolver/channelFlow/turbSolver.cpp -o CMakeFiles/reducedSolver.dir/turbSolver.cpp.s

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o.requires:

.PHONY : src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o.requires

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o.provides: src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o.requires
	$(MAKE) -f src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/build.make src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o.provides.build
.PHONY : src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o.provides

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o.provides.build: src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o


# Object files for target reducedSolver
reducedSolver_OBJECTS = \
"CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o" \
"CMakeFiles/reducedSolver.dir/turbSolver.cpp.o"

# External object files for target reducedSolver
reducedSolver_EXTERNAL_OBJECTS =

src/forwardSolver/channelFlow/libreducedSolver.a: src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o
src/forwardSolver/channelFlow/libreducedSolver.a: src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o
src/forwardSolver/channelFlow/libreducedSolver.a: src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/build.make
src/forwardSolver/channelFlow/libreducedSolver.a: src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libreducedSolver.a"
	cd /home/fenics/shared/src/forwardSolver/channelFlow && $(CMAKE_COMMAND) -P CMakeFiles/reducedSolver.dir/cmake_clean_target.cmake
	cd /home/fenics/shared/src/forwardSolver/channelFlow && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reducedSolver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/build: src/forwardSolver/channelFlow/libreducedSolver.a

.PHONY : src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/build

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/requires: src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/ChannelCppSolverBiMesh.cpp.o.requires
src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/requires: src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/turbSolver.cpp.o.requires

.PHONY : src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/requires

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/clean:
	cd /home/fenics/shared/src/forwardSolver/channelFlow && $(CMAKE_COMMAND) -P CMakeFiles/reducedSolver.dir/cmake_clean.cmake
.PHONY : src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/clean

src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/depend:
	cd /home/fenics/shared && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fenics/shared /home/fenics/shared/src/forwardSolver/channelFlow /home/fenics/shared /home/fenics/shared/src/forwardSolver/channelFlow /home/fenics/shared/src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/forwardSolver/channelFlow/CMakeFiles/reducedSolver.dir/depend

