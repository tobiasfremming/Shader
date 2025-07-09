# Hackerspace Shader Art OpenGL teplate

![Animation Example](dolphin.gif)



## What do i do?
Clone the repo
go the the res/shaders folder and create a new shader
change the current shader in use in gamelogic.cpp
Have fun
### Windows

Install Microsoft Visual Studio Express and CMake.
You may use CMake-gui or the command-line cmake to generate a Visual Studio solution.


### Build
```bash
cd shaderopengl
cmake --build build
```

### Run

```bash
cd shaderopengl/build/Debug
glowbox.exe

```


### Linux:

Make sure you have a C/C++ compiler such as  GCC, CMake and Git.

	make run

which is equivalent to

	git submodule update --init
	cd build
	cmake ..
	make
	./glowbox
