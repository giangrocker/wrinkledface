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
CMAKE_SOURCE_DIR = /home/giang-rocker/AniAge/eos_Mac

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/giang-rocker/AniAge/eos_Mac/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/generate-obj.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/generate-obj.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/generate-obj.dir/flags.make

examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o: examples/CMakeFiles/generate-obj.dir/flags.make
examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o: ../examples/generate-obj.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giang-rocker/AniAge/eos_Mac/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o"
	cd /home/giang-rocker/AniAge/eos_Mac/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/generate-obj.dir/generate-obj.cpp.o -c /home/giang-rocker/AniAge/eos_Mac/examples/generate-obj.cpp

examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/generate-obj.dir/generate-obj.cpp.i"
	cd /home/giang-rocker/AniAge/eos_Mac/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giang-rocker/AniAge/eos_Mac/examples/generate-obj.cpp > CMakeFiles/generate-obj.dir/generate-obj.cpp.i

examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/generate-obj.dir/generate-obj.cpp.s"
	cd /home/giang-rocker/AniAge/eos_Mac/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giang-rocker/AniAge/eos_Mac/examples/generate-obj.cpp -o CMakeFiles/generate-obj.dir/generate-obj.cpp.s

examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o.requires:

.PHONY : examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o.requires

examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o.provides: examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/generate-obj.dir/build.make examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o.provides.build
.PHONY : examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o.provides

examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o.provides.build: examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o


# Object files for target generate-obj
generate__obj_OBJECTS = \
"CMakeFiles/generate-obj.dir/generate-obj.cpp.o"

# External object files for target generate-obj
generate__obj_EXTERNAL_OBJECTS =

examples/generate-obj: examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o
examples/generate-obj: examples/CMakeFiles/generate-obj.dir/build.make
examples/generate-obj: /home/giang-rocker/anaconda3/lib/libopencv_calib3d.so.3.2.0
examples/generate-obj: /usr/lib/x86_64-linux-gnu/libboost_system.so
examples/generate-obj: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
examples/generate-obj: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
examples/generate-obj: /home/giang-rocker/anaconda3/lib/libopencv_features2d.so.3.2.0
examples/generate-obj: /home/giang-rocker/anaconda3/lib/libopencv_flann.so.3.2.0
examples/generate-obj: /home/giang-rocker/anaconda3/lib/libopencv_ml.so.3.2.0
examples/generate-obj: /home/giang-rocker/anaconda3/lib/libopencv_highgui.so.3.2.0
examples/generate-obj: /home/giang-rocker/anaconda3/lib/libopencv_videoio.so.3.2.0
examples/generate-obj: /home/giang-rocker/anaconda3/lib/libopencv_imgcodecs.so.3.2.0
examples/generate-obj: /home/giang-rocker/anaconda3/lib/libopencv_imgproc.so.3.2.0
examples/generate-obj: /home/giang-rocker/anaconda3/lib/libopencv_core.so.3.2.0
examples/generate-obj: examples/CMakeFiles/generate-obj.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/giang-rocker/AniAge/eos_Mac/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable generate-obj"
	cd /home/giang-rocker/AniAge/eos_Mac/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/generate-obj.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/generate-obj.dir/build: examples/generate-obj

.PHONY : examples/CMakeFiles/generate-obj.dir/build

examples/CMakeFiles/generate-obj.dir/requires: examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o.requires

.PHONY : examples/CMakeFiles/generate-obj.dir/requires

examples/CMakeFiles/generate-obj.dir/clean:
	cd /home/giang-rocker/AniAge/eos_Mac/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/generate-obj.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/generate-obj.dir/clean

examples/CMakeFiles/generate-obj.dir/depend:
	cd /home/giang-rocker/AniAge/eos_Mac/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/giang-rocker/AniAge/eos_Mac /home/giang-rocker/AniAge/eos_Mac/examples /home/giang-rocker/AniAge/eos_Mac/build /home/giang-rocker/AniAge/eos_Mac/build/examples /home/giang-rocker/AniAge/eos_Mac/build/examples/CMakeFiles/generate-obj.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/generate-obj.dir/depend

