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
include CMakeFiles/jialin_test2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/jialin_test2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/jialin_test2.dir/flags.make

CMakeFiles/jialin_test2.dir/jialin2.cpp.o: CMakeFiles/jialin_test2.dir/flags.make
CMakeFiles/jialin_test2.dir/jialin2.cpp.o: ../jialin2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jialin/Documents/VSC_Projects/test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/jialin_test2.dir/jialin2.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jialin_test2.dir/jialin2.cpp.o -c /home/jialin/Documents/VSC_Projects/test/jialin2.cpp

CMakeFiles/jialin_test2.dir/jialin2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jialin_test2.dir/jialin2.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jialin/Documents/VSC_Projects/test/jialin2.cpp > CMakeFiles/jialin_test2.dir/jialin2.cpp.i

CMakeFiles/jialin_test2.dir/jialin2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jialin_test2.dir/jialin2.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jialin/Documents/VSC_Projects/test/jialin2.cpp -o CMakeFiles/jialin_test2.dir/jialin2.cpp.s

CMakeFiles/jialin_test2.dir/jialin2.cpp.o.requires:

.PHONY : CMakeFiles/jialin_test2.dir/jialin2.cpp.o.requires

CMakeFiles/jialin_test2.dir/jialin2.cpp.o.provides: CMakeFiles/jialin_test2.dir/jialin2.cpp.o.requires
	$(MAKE) -f CMakeFiles/jialin_test2.dir/build.make CMakeFiles/jialin_test2.dir/jialin2.cpp.o.provides.build
.PHONY : CMakeFiles/jialin_test2.dir/jialin2.cpp.o.provides

CMakeFiles/jialin_test2.dir/jialin2.cpp.o.provides.build: CMakeFiles/jialin_test2.dir/jialin2.cpp.o


# Object files for target jialin_test2
jialin_test2_OBJECTS = \
"CMakeFiles/jialin_test2.dir/jialin2.cpp.o"

# External object files for target jialin_test2
jialin_test2_EXTERNAL_OBJECTS =

jialin_test2: CMakeFiles/jialin_test2.dir/jialin2.cpp.o
jialin_test2: CMakeFiles/jialin_test2.dir/build.make
jialin_test2: /usr/local/lib/libopencv_gapi.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_highgui.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_ml.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_objdetect.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_photo.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_stitching.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_video.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_videoio.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_dnn.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_calib3d.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_features2d.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_flann.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_imgproc.so.4.5.5
jialin_test2: /usr/local/lib/libopencv_core.so.4.5.5
jialin_test2: CMakeFiles/jialin_test2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jialin/Documents/VSC_Projects/test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable jialin_test2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/jialin_test2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/jialin_test2.dir/build: jialin_test2

.PHONY : CMakeFiles/jialin_test2.dir/build

CMakeFiles/jialin_test2.dir/requires: CMakeFiles/jialin_test2.dir/jialin2.cpp.o.requires

.PHONY : CMakeFiles/jialin_test2.dir/requires

CMakeFiles/jialin_test2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/jialin_test2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/jialin_test2.dir/clean

CMakeFiles/jialin_test2.dir/depend:
	cd /home/jialin/Documents/VSC_Projects/test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jialin/Documents/VSC_Projects/test /home/jialin/Documents/VSC_Projects/test /home/jialin/Documents/VSC_Projects/test/build /home/jialin/Documents/VSC_Projects/test/build /home/jialin/Documents/VSC_Projects/test/build/CMakeFiles/jialin_test2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/jialin_test2.dir/depend
