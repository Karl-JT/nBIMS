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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shared

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shared

# Include any dependencies generated for this target.
include src/tuts/stokes/CMakeFiles/stokeTutorial.dir/depend.make

# Include the progress variables for this target.
include src/tuts/stokes/CMakeFiles/stokeTutorial.dir/progress.make

# Include the compile flags for this target's objects.
include src/tuts/stokes/CMakeFiles/stokeTutorial.dir/flags.make

src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o: src/tuts/stokes/CMakeFiles/stokeTutorial.dir/flags.make
src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o: src/tuts/stokes/stokesTutorial2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o"
	cd /home/shared/src/tuts/stokes && /opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o -c /home/shared/src/tuts/stokes/stokesTutorial2.cpp

src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.i"
	cd /home/shared/src/tuts/stokes && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shared/src/tuts/stokes/stokesTutorial2.cpp > CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.i

src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.s"
	cd /home/shared/src/tuts/stokes && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shared/src/tuts/stokes/stokesTutorial2.cpp -o CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.s

src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o.requires:

.PHONY : src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o.requires

src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o.provides: src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o.requires
	$(MAKE) -f src/tuts/stokes/CMakeFiles/stokeTutorial.dir/build.make src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o.provides.build
.PHONY : src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o.provides

src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o.provides.build: src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o


# Object files for target stokeTutorial
stokeTutorial_OBJECTS = \
"CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o"

# External object files for target stokeTutorial
stokeTutorial_EXTERNAL_OBJECTS =

bin/stokeTutorial: src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o
bin/stokeTutorial: src/tuts/stokes/CMakeFiles/stokeTutorial.dir/build.make
bin/stokeTutorial: src/tool/numericalRecipes/libnumericalRecipes.a
bin/stokeTutorial: /usr/local/lib/libgsl.so
bin/stokeTutorial: /usr/local/lib/libgslcblas.so
bin/stokeTutorial: src/tuts/stokes/CMakeFiles/stokeTutorial.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/stokeTutorial"
	cd /home/shared/src/tuts/stokes && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stokeTutorial.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/tuts/stokes/CMakeFiles/stokeTutorial.dir/build: bin/stokeTutorial

.PHONY : src/tuts/stokes/CMakeFiles/stokeTutorial.dir/build

src/tuts/stokes/CMakeFiles/stokeTutorial.dir/requires: src/tuts/stokes/CMakeFiles/stokeTutorial.dir/stokesTutorial2.cpp.o.requires

.PHONY : src/tuts/stokes/CMakeFiles/stokeTutorial.dir/requires

src/tuts/stokes/CMakeFiles/stokeTutorial.dir/clean:
	cd /home/shared/src/tuts/stokes && $(CMAKE_COMMAND) -P CMakeFiles/stokeTutorial.dir/cmake_clean.cmake
.PHONY : src/tuts/stokes/CMakeFiles/stokeTutorial.dir/clean

src/tuts/stokes/CMakeFiles/stokeTutorial.dir/depend:
	cd /home/shared && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shared /home/shared/src/tuts/stokes /home/shared /home/shared/src/tuts/stokes /home/shared/src/tuts/stokes/CMakeFiles/stokeTutorial.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/tuts/stokes/CMakeFiles/stokeTutorial.dir/depend

