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
include examples/CMakeFiles/fit-model.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/fit-model.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/fit-model.dir/flags.make

examples/CMakeFiles/fit-model.dir/fit-model.cpp.o: examples/CMakeFiles/fit-model.dir/flags.make
examples/CMakeFiles/fit-model.dir/fit-model.cpp.o: ../examples/fit-model.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giang-rocker/AniAge/eos_Mac/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/fit-model.dir/fit-model.cpp.o"
	cd /home/giang-rocker/AniAge/eos_Mac/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fit-model.dir/fit-model.cpp.o -c /home/giang-rocker/AniAge/eos_Mac/examples/fit-model.cpp

examples/CMakeFiles/fit-model.dir/fit-model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fit-model.dir/fit-model.cpp.i"
	cd /home/giang-rocker/AniAge/eos_Mac/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giang-rocker/AniAge/eos_Mac/examples/fit-model.cpp > CMakeFiles/fit-model.dir/fit-model.cpp.i

examples/CMakeFiles/fit-model.dir/fit-model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fit-model.dir/fit-model.cpp.s"
	cd /home/giang-rocker/AniAge/eos_Mac/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giang-rocker/AniAge/eos_Mac/examples/fit-model.cpp -o CMakeFiles/fit-model.dir/fit-model.cpp.s

examples/CMakeFiles/fit-model.dir/fit-model.cpp.o.requires:

.PHONY : examples/CMakeFiles/fit-model.dir/fit-model.cpp.o.requires

examples/CMakeFiles/fit-model.dir/fit-model.cpp.o.provides: examples/CMakeFiles/fit-model.dir/fit-model.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/fit-model.dir/build.make examples/CMakeFiles/fit-model.dir/fit-model.cpp.o.provides.build
.PHONY : examples/CMakeFiles/fit-model.dir/fit-model.cpp.o.provides

examples/CMakeFiles/fit-model.dir/fit-model.cpp.o.provides.build: examples/CMakeFiles/fit-model.dir/fit-model.cpp.o


# Object files for target fit-model
fit__model_OBJECTS = \
"CMakeFiles/fit-model.dir/fit-model.cpp.o"

# External object files for target fit-model
fit__model_EXTERNAL_OBJECTS =

examples/fit-model: examples/CMakeFiles/fit-model.dir/fit-model.cpp.o
examples/fit-model: examples/CMakeFiles/fit-model.dir/build.make
examples/fit-model: /home/giang-rocker/anaconda3/lib/libopencv_calib3d.so.3.2.0
examples/fit-model: /usr/lib/x86_64-linux-gnu/libboost_system.so
examples/fit-model: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
examples/fit-model: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
examples/fit-model: /home/giang-rocker/anaconda3/lib/libopencv_features2d.so.3.2.0
examples/fit-model: /home/giang-rocker/anaconda3/lib/libopencv_flann.so.3.2.0
examples/fit-model: /home/giang-rocker/anaconda3/lib/libopencv_ml.so.3.2.0
examples/fit-model: /home/giang-rocker/anaconda3/lib/libopencv_highgui.so.3.2.0
examples/fit-model: /home/giang-rocker/anaconda3/lib/libopencv_videoio.so.3.2.0
examples/fit-model: /home/giang-rocker/anaconda3/lib/libopencv_imgcodecs.so.3.2.0
examples/fit-model: /home/giang-rocker/anaconda3/lib/libopencv_imgproc.so.3.2.0
examples/fit-model: /home/giang-rocker/anaconda3/lib/libopencv_core.so.3.2.0
examples/fit-model: examples/CMakeFiles/fit-model.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/giang-rocker/AniAge/eos_Mac/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fit-model"
	cd /home/giang-rocker/AniAge/eos_Mac/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fit-model.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/fit-model.dir/build: examples/fit-model

.PHONY : examples/CMakeFiles/fit-model.dir/build

examples/CMakeFiles/fit-model.dir/requires: examples/CMakeFiles/fit-model.dir/fit-model.cpp.o.requires

.PHONY : examples/CMakeFiles/fit-model.dir/requires

examples/CMakeFiles/fit-model.dir/clean:
	cd /home/giang-rocker/AniAge/eos_Mac/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/fit-model.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/fit-model.dir/clean

examples/CMakeFiles/fit-model.dir/depend:
	cd /home/giang-rocker/AniAge/eos_Mac/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/giang-rocker/AniAge/eos_Mac /home/giang-rocker/AniAge/eos_Mac/examples /home/giang-rocker/AniAge/eos_Mac/build /home/giang-rocker/AniAge/eos_Mac/build/examples /home/giang-rocker/AniAge/eos_Mac/build/examples/CMakeFiles/fit-model.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/fit-model.dir/depend

