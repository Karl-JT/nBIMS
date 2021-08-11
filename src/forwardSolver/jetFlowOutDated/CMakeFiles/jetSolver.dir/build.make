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
include src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/depend.make

# Include the progress variables for this target.
include src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/progress.make

# Include the compile flags for this target's objects.
include src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/flags.make

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o: src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/flags.make
src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o: src/forwardSolver/jetFlow/forwardSolver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/share/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o"
	cd /home/fenics/share/src/forwardSolver/jetFlow && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jetSolver.dir/forwardSolver.cpp.o -c /home/fenics/share/src/forwardSolver/jetFlow/forwardSolver.cpp

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jetSolver.dir/forwardSolver.cpp.i"
	cd /home/fenics/share/src/forwardSolver/jetFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/share/src/forwardSolver/jetFlow/forwardSolver.cpp > CMakeFiles/jetSolver.dir/forwardSolver.cpp.i

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jetSolver.dir/forwardSolver.cpp.s"
	cd /home/fenics/share/src/forwardSolver/jetFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/share/src/forwardSolver/jetFlow/forwardSolver.cpp -o CMakeFiles/jetSolver.dir/forwardSolver.cpp.s

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o.requires:

.PHONY : src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o.requires

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o.provides: src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o.requires
	$(MAKE) -f src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/build.make src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o.provides.build
.PHONY : src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o.provides

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o.provides.build: src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o


src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o: src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/flags.make
src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o: src/forwardSolver/jetFlow/jetFlowSolver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/share/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o"
	cd /home/fenics/share/src/forwardSolver/jetFlow && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o -c /home/fenics/share/src/forwardSolver/jetFlow/jetFlowSolver.cpp

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.i"
	cd /home/fenics/share/src/forwardSolver/jetFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/share/src/forwardSolver/jetFlow/jetFlowSolver.cpp > CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.i

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.s"
	cd /home/fenics/share/src/forwardSolver/jetFlow && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/share/src/forwardSolver/jetFlow/jetFlowSolver.cpp -o CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.s

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o.requires:

.PHONY : src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o.requires

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o.provides: src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o.requires
	$(MAKE) -f src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/build.make src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o.provides.build
.PHONY : src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o.provides

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o.provides.build: src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o


# Object files for target jetSolver
jetSolver_OBJECTS = \
"CMakeFiles/jetSolver.dir/forwardSolver.cpp.o" \
"CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o"

# External object files for target jetSolver
jetSolver_EXTERNAL_OBJECTS =

src/forwardSolver/jetFlow/libjetSolver.a: src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o
src/forwardSolver/jetFlow/libjetSolver.a: src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o
src/forwardSolver/jetFlow/libjetSolver.a: src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/build.make
src/forwardSolver/jetFlow/libjetSolver.a: src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fenics/share/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libjetSolver.a"
	cd /home/fenics/share/src/forwardSolver/jetFlow && $(CMAKE_COMMAND) -P CMakeFiles/jetSolver.dir/cmake_clean_target.cmake
	cd /home/fenics/share/src/forwardSolver/jetFlow && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/jetSolver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/build: src/forwardSolver/jetFlow/libjetSolver.a

.PHONY : src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/build

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/requires: src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/forwardSolver.cpp.o.requires
src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/requires: src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/jetFlowSolver.cpp.o.requires

.PHONY : src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/requires

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/clean:
	cd /home/fenics/share/src/forwardSolver/jetFlow && $(CMAKE_COMMAND) -P CMakeFiles/jetSolver.dir/cmake_clean.cmake
.PHONY : src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/clean

src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/depend:
	cd /home/fenics/share && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fenics/share /home/fenics/share/src/forwardSolver/jetFlow /home/fenics/share /home/fenics/share/src/forwardSolver/jetFlow /home/fenics/share/src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/forwardSolver/jetFlow/CMakeFiles/jetSolver.dir/depend
