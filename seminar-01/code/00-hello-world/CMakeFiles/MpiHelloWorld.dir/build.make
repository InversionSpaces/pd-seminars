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
CMAKE_SOURCE_DIR = /home/01-mpi/00-hello-world

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/01-mpi/00-hello-world

# Include any dependencies generated for this target.
include CMakeFiles/MpiHelloWorld.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/MpiHelloWorld.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MpiHelloWorld.dir/flags.make

CMakeFiles/MpiHelloWorld.dir/main.cpp.o: CMakeFiles/MpiHelloWorld.dir/flags.make
CMakeFiles/MpiHelloWorld.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/01-mpi/00-hello-world/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MpiHelloWorld.dir/main.cpp.o"
	/usr/bin/mpic++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MpiHelloWorld.dir/main.cpp.o -c /home/01-mpi/00-hello-world/main.cpp

CMakeFiles/MpiHelloWorld.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MpiHelloWorld.dir/main.cpp.i"
	/usr/bin/mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/01-mpi/00-hello-world/main.cpp > CMakeFiles/MpiHelloWorld.dir/main.cpp.i

CMakeFiles/MpiHelloWorld.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MpiHelloWorld.dir/main.cpp.s"
	/usr/bin/mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/01-mpi/00-hello-world/main.cpp -o CMakeFiles/MpiHelloWorld.dir/main.cpp.s

CMakeFiles/MpiHelloWorld.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/MpiHelloWorld.dir/main.cpp.o.requires

CMakeFiles/MpiHelloWorld.dir/main.cpp.o.provides: CMakeFiles/MpiHelloWorld.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/MpiHelloWorld.dir/build.make CMakeFiles/MpiHelloWorld.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/MpiHelloWorld.dir/main.cpp.o.provides

CMakeFiles/MpiHelloWorld.dir/main.cpp.o.provides.build: CMakeFiles/MpiHelloWorld.dir/main.cpp.o


# Object files for target MpiHelloWorld
MpiHelloWorld_OBJECTS = \
"CMakeFiles/MpiHelloWorld.dir/main.cpp.o"

# External object files for target MpiHelloWorld
MpiHelloWorld_EXTERNAL_OBJECTS =

bin/MpiHelloWorld: CMakeFiles/MpiHelloWorld.dir/main.cpp.o
bin/MpiHelloWorld: CMakeFiles/MpiHelloWorld.dir/build.make
bin/MpiHelloWorld: CMakeFiles/MpiHelloWorld.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/01-mpi/00-hello-world/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/MpiHelloWorld"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MpiHelloWorld.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MpiHelloWorld.dir/build: bin/MpiHelloWorld

.PHONY : CMakeFiles/MpiHelloWorld.dir/build

CMakeFiles/MpiHelloWorld.dir/requires: CMakeFiles/MpiHelloWorld.dir/main.cpp.o.requires

.PHONY : CMakeFiles/MpiHelloWorld.dir/requires

CMakeFiles/MpiHelloWorld.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MpiHelloWorld.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MpiHelloWorld.dir/clean

CMakeFiles/MpiHelloWorld.dir/depend:
	cd /home/01-mpi/00-hello-world && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/01-mpi/00-hello-world /home/01-mpi/00-hello-world /home/01-mpi/00-hello-world /home/01-mpi/00-hello-world /home/01-mpi/00-hello-world/CMakeFiles/MpiHelloWorld.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MpiHelloWorld.dir/depend

