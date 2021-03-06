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
CMAKE_BINARY_DIR = /home/fenics/shared/build

# Include any dependencies generated for this target.
include CMakeFiles/jetFlowTest.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/jetFlowTest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/jetFlowTest.dir/flags.make

CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o: CMakeFiles/jetFlowTest.dir/flags.make
CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o: ../jetFlowTest.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fenics/shared/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o -c /home/fenics/shared/jetFlowTest.cpp

CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fenics/shared/jetFlowTest.cpp > CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.i

CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fenics/shared/jetFlowTest.cpp -o CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.s

CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o.requires:

.PHONY : CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o.requires

CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o.provides: CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o.requires
	$(MAKE) -f CMakeFiles/jetFlowTest.dir/build.make CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o.provides.build
.PHONY : CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o.provides

CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o.provides.build: CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o


# Object files for target jetFlowTest
jetFlowTest_OBJECTS = \
"CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o"

# External object files for target jetFlowTest
jetFlowTest_EXTERNAL_OBJECTS =

jetFlowTest: CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o
jetFlowTest: CMakeFiles/jetFlowTest.dir/build.make
jetFlowTest: jetFlowSolver/libjetFlowSolver.a
jetFlowTest: /usr/local/lib/libdolfin.so.2019.1.0
jetFlowTest: /usr/lib/x86_64-linux-gnu/libboost_timer.so
jetFlowTest: /usr/lib/x86_64-linux-gnu/hdf5/mpich/libhdf5.so
jetFlowTest: /usr/lib/x86_64-linux-gnu/libsz.so
jetFlowTest: /usr/lib/x86_64-linux-gnu/libz.so
jetFlowTest: /usr/lib/x86_64-linux-gnu/libdl.so
jetFlowTest: /usr/lib/x86_64-linux-gnu/libm.so
jetFlowTest: /usr/local/slepc-32/lib/libslepc.so
jetFlowTest: /usr/local/petsc-32/lib/libpetsc.so
jetFlowTest: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
jetFlowTest: /usr/lib/x86_64-linux-gnu/libmpich.so
jetFlowTest: CMakeFiles/jetFlowTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fenics/shared/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable jetFlowTest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/jetFlowTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/jetFlowTest.dir/build: jetFlowTest

.PHONY : CMakeFiles/jetFlowTest.dir/build

CMakeFiles/jetFlowTest.dir/requires: CMakeFiles/jetFlowTest.dir/jetFlowTest.cpp.o.requires

.PHONY : CMakeFiles/jetFlowTest.dir/requires

CMakeFiles/jetFlowTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/jetFlowTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/jetFlowTest.dir/clean

CMakeFiles/jetFlowTest.dir/depend:
	cd /home/fenics/shared/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fenics/shared /home/fenics/shared /home/fenics/shared/build /home/fenics/shared/build /home/fenics/shared/build/CMakeFiles/jetFlowTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/jetFlowTest.dir/depend

