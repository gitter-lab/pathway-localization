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
include CMakeFiles/Demo_Train.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Demo_Train.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Demo_Train.dir/flags.make

CMakeFiles/Demo_Train.dir/Demo_Train.o: CMakeFiles/Demo_Train.dir/flags.make
CMakeFiles/Demo_Train.dir/Demo_Train.o: ../Demo\ Train.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Demo_Train.dir/Demo_Train.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Demo_Train.dir/Demo_Train.o -c "/home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/Demo Train.cpp"

CMakeFiles/Demo_Train.dir/Demo_Train.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Demo_Train.dir/Demo_Train.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/Demo Train.cpp" > CMakeFiles/Demo_Train.dir/Demo_Train.i

CMakeFiles/Demo_Train.dir/Demo_Train.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Demo_Train.dir/Demo_Train.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/Demo Train.cpp" -o CMakeFiles/Demo_Train.dir/Demo_Train.s

CMakeFiles/Demo_Train.dir/Demo_Train.o.requires:

.PHONY : CMakeFiles/Demo_Train.dir/Demo_Train.o.requires

CMakeFiles/Demo_Train.dir/Demo_Train.o.provides: CMakeFiles/Demo_Train.dir/Demo_Train.o.requires
	$(MAKE) -f CMakeFiles/Demo_Train.dir/build.make CMakeFiles/Demo_Train.dir/Demo_Train.o.provides.build
.PHONY : CMakeFiles/Demo_Train.dir/Demo_Train.o.provides

CMakeFiles/Demo_Train.dir/Demo_Train.o.provides.build: CMakeFiles/Demo_Train.dir/Demo_Train.o


# Object files for target Demo_Train
Demo_Train_OBJECTS = \
"CMakeFiles/Demo_Train.dir/Demo_Train.o"

# External object files for target Demo_Train
Demo_Train_EXTERNAL_OBJECTS =

Demo\ Train: CMakeFiles/Demo_Train.dir/Demo_Train.o
Demo\ Train: CMakeFiles/Demo_Train.dir/build.make
Demo\ Train: CMakeFiles/Demo_Train.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable \"Demo Train\""
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Demo_Train.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Demo_Train.dir/build: Demo\ Train

.PHONY : CMakeFiles/Demo_Train.dir/build

CMakeFiles/Demo_Train.dir/requires: CMakeFiles/Demo_Train.dir/Demo_Train.o.requires

.PHONY : CMakeFiles/Demo_Train.dir/requires

CMakeFiles/Demo_Train.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Demo_Train.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Demo_Train.dir/clean

CMakeFiles/Demo_Train.dir/depend:
	cd /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build /home/chris/research/pathwayLocalization/scripts/exploratoryScripts/dgm/build/CMakeFiles/Demo_Train.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Demo_Train.dir/depend
