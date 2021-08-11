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
include src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/depend.make

# Include the progress variables for this target.
include src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/progress.make

# Include the compile flags for this target's objects.
include src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/flags.make

# Object files for target stochasticNewtonMCMC
stochasticNewtonMCMC_OBJECTS =

# External object files for target stochasticNewtonMCMC
stochasticNewtonMCMC_EXTERNAL_OBJECTS =

src/mcmcVersions/stochasticNewtonMCMC/libstochasticNewtonMCMC.a: src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/build.make
src/mcmcVersions/stochasticNewtonMCMC/libstochasticNewtonMCMC.a: src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fenics/shared/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library libstochasticNewtonMCMC.a"
	cd /home/fenics/shared/src/mcmcVersions/stochasticNewtonMCMC && $(CMAKE_COMMAND) -P CMakeFiles/stochasticNewtonMCMC.dir/cmake_clean_target.cmake
	cd /home/fenics/shared/src/mcmcVersions/stochasticNewtonMCMC && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stochasticNewtonMCMC.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/build: src/mcmcVersions/stochasticNewtonMCMC/libstochasticNewtonMCMC.a

.PHONY : src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/build

src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/requires:

.PHONY : src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/requires

src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/clean:
	cd /home/fenics/shared/src/mcmcVersions/stochasticNewtonMCMC && $(CMAKE_COMMAND) -P CMakeFiles/stochasticNewtonMCMC.dir/cmake_clean.cmake
.PHONY : src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/clean

src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/depend:
	cd /home/fenics/shared && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fenics/shared /home/fenics/shared/src/mcmcVersions/stochasticNewtonMCMC /home/fenics/shared /home/fenics/shared/src/mcmcVersions/stochasticNewtonMCMC /home/fenics/shared/src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/mcmcVersions/stochasticNewtonMCMC/CMakeFiles/stochasticNewtonMCMC.dir/depend
