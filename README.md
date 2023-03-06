# program-evolution
Experimental system to evolve programs to solve problems.

This is a small personal project that uses an evolutionary algorithm to evolve program code to solve grid-based problems. It uses CUDA to run the programs faster. To simplify compilation, all files are header-only, including dependencies.

The programs it generates run on a regular 2D grid. Each cell in the grid has one more read-only inputs, write-only outputs and read/writeable variables. Grid cells can also read from any of the 8 neighbouring cell inputs and variables. In addition, each program has a table of read-only constant values. Otherwise, the program uses a stack to store intermediate results. For example, an add operation instruction pops the top 2 items off the stack and then pushes the result back onto the stack.

Presently, only one boundary condition is supported - a value of 0 is returned when a cell on the edge of the grid attempts to read from an input or variable neighbour that is out of bounds.

Right now, the evolutionary algorithm is simple, featuring just a single listing where programs compete. At each generation, the top N scoring programs survive to stay in the listing and to generate candidates for the next generation. A simple combination and mutation procedure is implemented. Periodically, the top scoring programs are reported in JSON format. 

The demo application attempts to find a program that can blur the 4 cardinal neighbouring cells such that:
Output = (Top + Bottom + Right + Left) / 4

It is structured such that it runs the program against a series of training cases with input and output examples. The training goal (score function) attempts to minimize the L1 distance between the program's output and the examples' output. Variables are cleared before running each training case.

Although the demo only runs programs once for each training case, variables can persist their values across runs of the same program. This can be useful for simulation scenarios where the programs are run in a loop, computing the next state. For example, the variables might encode useful hidden parameters in a fluid simulation.

Usage:
- Install latest CUDA on your system
- ./build.bat (tested on windows but should be trivial to modify to run on unix platforms)
- bin/program-evolution.exe for demo application
- bin/test-program-device.exe for unit tests

Overview:
main.cu: runs evolution in a demo application.
test-program-device.cu: unit tests for running the program on the device (CUDA).
doctest.h, json.h: header-only dependencies (credited below).
util/: miscellaneous useful utility classes and functions.
evolution/: handles the evolutionary algorithm.
program/: manages programs, how to mutate them, and how to run them on the device.

Future Work:
- Simulate isolated biomes, ecological niches, etc. to promote diversity.
- Ability to stop and resume.
- Support more boundary conditions.
- More mutation operators (such as translocations and duplications).
- Experiment with using shared memory and advanced CUDA operators in program emulator kernel.
- Experiment with parameters to achieve faster convergence.
- Investigate whether CPU processes are a bottleneck and if multi-threading is needed.
- Refactor API further and simplify application developer experience.
- Unit Tests for program mutation and evolutionary algorithm.

Direct External Credits:
https://github.com/nlohmann/json: for writing out the evolving programs in JSON format.
https://github.com/doctest/doctest: for unit tests.
