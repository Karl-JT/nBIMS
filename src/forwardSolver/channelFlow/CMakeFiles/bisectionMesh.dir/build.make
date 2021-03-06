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
include src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/depend.make

# Include the progress variables for this target.
include src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/progress.make

# Include the compile flags for this target's objects.
include src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/flags.make

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o: src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/flags.make
src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o: src/forwardSolver/bisectionMesh/ChannelCppSolverBiMesh.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o"
	cd /home/fenics/shared/src/forwardSolver/bisectionMesh && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o -c /home/fenics/shared/src/forwardSolver/bisectionMesh/ChannelCppSolverBiMesh.cpp

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.i"
	cd /home/fenics/shared/src/forwardSolver/bisectionMesh && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/shared/src/forwardSolver/bisectionMesh/ChannelCppSolverBiMesh.cpp > CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.i

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.s"
	cd /home/fenics/shared/src/forwardSolver/bisectionMesh && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/shared/src/forwardSolver/bisectionMesh/ChannelCppSolverBiMesh.cpp -o CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.s

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o.requires:

.PHONY : src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o.requires

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o.provides: src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o.requires
	$(MAKE) -f src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/build.make src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o.provides.build
.PHONY : src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o.provides

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o.provides.build: src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o


src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o: src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/flags.make
src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o: src/forwardSolver/bisectionMesh/turbSolver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o"
	cd /home/fenics/shared/src/forwardSolver/bisectionMesh && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o -c /home/fenics/shared/src/forwardSolver/bisectionMesh/turbSolver.cpp

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bisectionMesh.dir/turbSolver.cpp.i"
	cd /home/fenics/shared/src/forwardSolver/bisectionMesh && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/shared/src/forwardSolver/bisectionMesh/turbSolver.cpp > CMakeFiles/bisectionMesh.dir/turbSolver.cpp.i

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bisectionMesh.dir/turbSolver.cpp.s"
	cd /home/fenics/shared/src/forwardSolver/bisectionMesh && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/shared/src/forwardSolver/bisectionMesh/turbSolver.cpp -o CMakeFiles/bisectionMesh.dir/turbSolver.cpp.s

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o.requires:

.PHONY : src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o.requires

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o.provides: src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o.requires
	$(MAKE) -f src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/build.make src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o.provides.build
.PHONY : src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o.provides

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o.provides.build: src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o


# Object files for target bisectionMesh
bisectionMesh_OBJECTS = \
"CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o" \
"CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o"

# External object files for target bisectionMesh
bisectionMesh_EXTERNAL_OBJECTS =

src/forwardSolver/bisectionMesh/libbisectionMesh.a: src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o
src/forwardSolver/bisectionMesh/libbisectionMesh.a: src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o
src/forwardSolver/bisectionMesh/libbisectionMesh.a: src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/build.make
src/forwardSolver/bisectionMesh/libbisectionMesh.a: src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libbisectionMesh.a"
	cd /home/fenics/shared/src/forwardSolver/bisectionMesh && $(CMAKE_COMMAND) -P CMakeFiles/bisectionMesh.dir/cmake_clean_target.cmake
	cd /home/fenics/shared/src/forwardSolver/bisectionMesh && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bisectionMesh.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/build: src/forwardSolver/bisectionMesh/libbisectionMesh.a

.PHONY : src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/build

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/requires: src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/ChannelCppSolverBiMesh.cpp.o.requires
src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/requires: src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/turbSolver.cpp.o.requires

.PHONY : src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/requires

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/clean:
	cd /home/fenics/shared/src/forwardSolver/bisectionMesh && $(CMAKE_COMMAND) -P CMakeFiles/bisectionMesh.dir/cmake_clean.cmake
.PHONY : src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/clean

src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/depend:
	cd /home/fenics/shared && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fenics/shared /home/fenics/shared/src/forwardSolver/bisectionMesh /home/fenics/shared /home/fenics/shared/src/forwardSolver/bisectionMesh /home/fenics/shared/src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/forwardSolver/bisectionMesh/CMakeFiles/bisectionMesh.dir/depend

