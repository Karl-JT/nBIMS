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
include src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/depend.make

# Include the progress variables for this target.
include src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/progress.make

# Include the compile flags for this target's objects.
include src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/flags.make

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o: src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/flags.make
src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o: src/forwardSolver/cavityFlow/cavityFlowCase.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o"
	cd /home/fenics/shared/src/forwardSolver/cavityFlow && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o -c /home/fenics/shared/src/forwardSolver/cavityFlow/cavityFlowCase.cpp

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.i"
	cd /home/fenics/shared/src/forwardSolver/cavityFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/shared/src/forwardSolver/cavityFlow/cavityFlowCase.cpp > CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.i

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.s"
	cd /home/fenics/shared/src/forwardSolver/cavityFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/shared/src/forwardSolver/cavityFlow/cavityFlowCase.cpp -o CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.s

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o.requires:

.PHONY : src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o.requires

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o.provides: src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o.requires
	$(MAKE) -f src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/build.make src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o.provides.build
.PHONY : src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o.provides

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o.provides.build: src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o


src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o: src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/flags.make
src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o: src/forwardSolver/cavityFlow/cavityFlowSolver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o"
	cd /home/fenics/shared/src/forwardSolver/cavityFlow && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o -c /home/fenics/shared/src/forwardSolver/cavityFlow/cavityFlowSolver.cpp

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.i"
	cd /home/fenics/shared/src/forwardSolver/cavityFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/shared/src/forwardSolver/cavityFlow/cavityFlowSolver.cpp > CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.i

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.s"
	cd /home/fenics/shared/src/forwardSolver/cavityFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/shared/src/forwardSolver/cavityFlow/cavityFlowSolver.cpp -o CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.s

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o.requires:

.PHONY : src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o.requires

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o.provides: src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o.requires
	$(MAKE) -f src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/build.make src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o.provides.build
.PHONY : src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o.provides

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o.provides.build: src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o


# Object files for target cavityFlow
cavityFlow_OBJECTS = \
"CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o" \
"CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o"

# External object files for target cavityFlow
cavityFlow_EXTERNAL_OBJECTS =

src/forwardSolver/cavityFlow/libcavityFlow.a: src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o
src/forwardSolver/cavityFlow/libcavityFlow.a: src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o
src/forwardSolver/cavityFlow/libcavityFlow.a: src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/build.make
src/forwardSolver/cavityFlow/libcavityFlow.a: src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libcavityFlow.a"
	cd /home/fenics/shared/src/forwardSolver/cavityFlow && $(CMAKE_COMMAND) -P CMakeFiles/cavityFlow.dir/cmake_clean_target.cmake
	cd /home/fenics/shared/src/forwardSolver/cavityFlow && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cavityFlow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/build: src/forwardSolver/cavityFlow/libcavityFlow.a

.PHONY : src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/build

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/requires: src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowCase.cpp.o.requires
src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/requires: src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/cavityFlowSolver.cpp.o.requires

.PHONY : src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/requires

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/clean:
	cd /home/fenics/shared/src/forwardSolver/cavityFlow && $(CMAKE_COMMAND) -P CMakeFiles/cavityFlow.dir/cmake_clean.cmake
.PHONY : src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/clean

src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/depend:
	cd /home/fenics/shared && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fenics/shared /home/fenics/shared/src/forwardSolver/cavityFlow /home/fenics/shared /home/fenics/shared/src/forwardSolver/cavityFlow /home/fenics/shared/src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/forwardSolver/cavityFlow/CMakeFiles/cavityFlow.dir/depend

