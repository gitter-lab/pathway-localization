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
CMAKE_SOURCE_DIR = /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build

# Include any dependencies generated for this target.
include CMakeFiles/Demo_Stereo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Demo_Stereo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Demo_Stereo.dir/flags.make

CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o: CMakeFiles/Demo_Stereo.dir/flags.make
CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o: ../Demo\ Stereo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o -c "/home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/Demo Stereo.cpp"

CMakeFiles/Demo_Stereo.dir/Demo_Stereo.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Demo_Stereo.dir/Demo_Stereo.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/Demo Stereo.cpp" > CMakeFiles/Demo_Stereo.dir/Demo_Stereo.i

CMakeFiles/Demo_Stereo.dir/Demo_Stereo.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Demo_Stereo.dir/Demo_Stereo.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/Demo Stereo.cpp" -o CMakeFiles/Demo_Stereo.dir/Demo_Stereo.s

CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o.requires:

.PHONY : CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o.requires

CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o.provides: CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o.requires
	$(MAKE) -f CMakeFiles/Demo_Stereo.dir/build.make CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o.provides.build
.PHONY : CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o.provides

CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o.provides.build: CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o


# Object files for target Demo_Stereo
Demo_Stereo_OBJECTS = \
"CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o"

# External object files for target Demo_Stereo
Demo_Stereo_EXTERNAL_OBJECTS =

Demo\ Stereo: CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o
Demo\ Stereo: CMakeFiles/Demo_Stereo.dir/build.make
Demo\ Stereo: CMakeFiles/Demo_Stereo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable \"Demo Stereo\""
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Demo_Stereo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Demo_Stereo.dir/build: Demo\ Stereo

.PHONY : CMakeFiles/Demo_Stereo.dir/build

CMakeFiles/Demo_Stereo.dir/requires: CMakeFiles/Demo_Stereo.dir/Demo_Stereo.o.requires

.PHONY : CMakeFiles/Demo_Stereo.dir/requires

CMakeFiles/Demo_Stereo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Demo_Stereo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Demo_Stereo.dir/clean

CMakeFiles/Demo_Stereo.dir/depend:
	cd /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build/CMakeFiles/Demo_Stereo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Demo_Stereo.dir/depend
