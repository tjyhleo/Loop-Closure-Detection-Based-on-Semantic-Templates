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
CMAKE_SOURCE_DIR = /home/jialin/Documents/VSC_Projects/test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jialin/Documents/VSC_Projects/test/build

# Include any dependencies generated for this target.
include CMakeFiles/test_cc.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_cc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_cc.dir/flags.make

CMakeFiles/test_cc.dir/test.cc.o: CMakeFiles/test_cc.dir/flags.make
CMakeFiles/test_cc.dir/test.cc.o: ../test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jialin/Documents/VSC_Projects/test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_cc.dir/test.cc.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_cc.dir/test.cc.o -c /home/jialin/Documents/VSC_Projects/test/test.cc

CMakeFiles/test_cc.dir/test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_cc.dir/test.cc.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jialin/Documents/VSC_Projects/test/test.cc > CMakeFiles/test_cc.dir/test.cc.i

CMakeFiles/test_cc.dir/test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_cc.dir/test.cc.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jialin/Documents/VSC_Projects/test/test.cc -o CMakeFiles/test_cc.dir/test.cc.s

CMakeFiles/test_cc.dir/test.cc.o.requires:

.PHONY : CMakeFiles/test_cc.dir/test.cc.o.requires

CMakeFiles/test_cc.dir/test.cc.o.provides: CMakeFiles/test_cc.dir/test.cc.o.requires
	$(MAKE) -f CMakeFiles/test_cc.dir/build.make CMakeFiles/test_cc.dir/test.cc.o.provides.build
.PHONY : CMakeFiles/test_cc.dir/test.cc.o.provides

CMakeFiles/test_cc.dir/test.cc.o.provides.build: CMakeFiles/test_cc.dir/test.cc.o


# Object files for target test_cc
test_cc_OBJECTS = \
"CMakeFiles/test_cc.dir/test.cc.o"

# External object files for target test_cc
test_cc_EXTERNAL_OBJECTS =

test_cc: CMakeFiles/test_cc.dir/test.cc.o
test_cc: CMakeFiles/test_cc.dir/build.make
test_cc: CMakeFiles/test_cc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jialin/Documents/VSC_Projects/test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_cc"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_cc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_cc.dir/build: test_cc

.PHONY : CMakeFiles/test_cc.dir/build

CMakeFiles/test_cc.dir/requires: CMakeFiles/test_cc.dir/test.cc.o.requires

.PHONY : CMakeFiles/test_cc.dir/requires

CMakeFiles/test_cc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_cc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_cc.dir/clean

CMakeFiles/test_cc.dir/depend:
	cd /home/jialin/Documents/VSC_Projects/test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jialin/Documents/VSC_Projects/test /home/jialin/Documents/VSC_Projects/test /home/jialin/Documents/VSC_Projects/test/build /home/jialin/Documents/VSC_Projects/test/build /home/jialin/Documents/VSC_Projects/test/build/CMakeFiles/test_cc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_cc.dir/depend
