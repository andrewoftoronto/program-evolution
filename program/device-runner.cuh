#pragma once
#include "program.cuh"
#include "util/cross-buffer.cuh"
#include "util/util.cuh"
#include <cassert>
#include <optional>

/**
* @brief Runs the given instruction.
* 
* @param instruction the instruction to run.
* @param x x-coordinate in the grid.
* @param y y-coordinate in the grid. 
* @param gridSizeX size of the grid along the x-axis.
* @param gridSizeY size of the grid along the y-axis.
* @param constants array of constants used by this program.
* @param inputs [I, Y, X] array of inputs for this program.
* @param variables [V, Y, X] array of variables for this program.
* @param outputs [O, Y, X] array of outputs for this program.
* @param stack array of current stack values.
*/
__device__ void runInstruction(const Instruction& instruction, 
        unsigned int x, unsigned int y, 
        unsigned int gridSizeX, unsigned int gridSizeY, 
        const float* constants, const float* inputs, float* variables, 
        float* outputs, Array<float>& stack) {

    // Number to add to prevent divide by 0 in programs.
    const float ZERO_THRESHOLD = 0.00001f;

    if (instruction.kind.isLoad()) {

        float value;
        if (instruction.kind == InstructionKind::LOAD_CONSTANT) {
            value = constants[instruction.argument];
        } else {

            // Variable and Input loads can take from neighbour cells; thus
            // there's 9 possible positions to take from for each channel.
            const float* tensor = instruction.kind == InstructionKind::LOAD_VARIABLE ? variables : inputs;
            unsigned int channel = instruction.argument / 9;
            unsigned int yRead = y + (instruction.argument % 9) / 3 - 1;
            unsigned int xRead = x + instruction.argument % 3 - 1;
            unsigned int readIndex = channel * gridSizeY * gridSizeX + yRead * gridSizeX + xRead;

            // Default any out of bounds read to 0. Note that negative
            // coordinates would wrap around since we're using unsigned.
            bool valid = xRead < gridSizeX && yRead < gridSizeY;
            if (valid) {
                value = tensor[readIndex];
            } else {
                value = 0;
            }
        }

        stack[stack.length] = value;
        stack.length++;
    } else if (instruction.kind.isStore()) {

        // If stack is empty, default to storing 0.
        float value;
        if (stack.length > 0) {
            value = stack[stack.length - 1];
            stack.length--;
        } else {
            value = 0;
        }

        // Unlike load, we can only store inside this cell of the grid.
        // instruction.argument indexes both variables and outputs, so here we
        // figure out which is meant.
        float* tensor;
        if (instruction.kind == InstructionKind::STORE_VARIABLE) {
            tensor = variables;
        } else {
            tensor = outputs;
        }
        tensor[instruction.argument * gridSizeY * gridSizeX + y * gridSizeX + x] = value;

    } else if (instruction.kind.isUnary()) {

        // If stack is empty, default to the argument being 0 instead.
        float a;
        if (stack.length > 0) {
            a = stack[stack.length - 1];

            // Duplicate by preventing the source copy from being removed.
            if (instruction.kind != InstructionKind::DUPLICATE) {
                stack.length--;
            }
        } else {
            a = 0;
        }

        float result;
        if (instruction.kind == InstructionKind::GT_0) {
            result = a > 0;
        } else if (instruction.kind == InstructionKind::RECIPRICOL) {
            result = a > ZERO_THRESHOLD ? 1 / a : 1 / (a + sign(a) * ZERO_THRESHOLD);
        } else if (instruction.kind == InstructionKind::ABS) {
            result = abs(a);
        } else {
            result = a;
        }
        stack[stack.length] = result;
        stack.length++;
        
    } else if (instruction.kind.isBinary()) {

        // If stack is empty, default to the argument being 0 instead.
        float a;
        if (stack.length > 0) {
            a = stack[stack.length - 1];
            stack.length--;
        } else {
            a = 0;
        }

        float b;
        if (stack.length > 0) {
            b = stack[stack.length - 1];
            stack.length--;
        } else {
            b = 0;
        }

        float result;
        if (instruction.kind == InstructionKind::ADD) {
            result = a + b;
        } else if (instruction.kind == InstructionKind::SUB) {
            result = a - b;
        } else if (instruction.kind == InstructionKind::MULTIPLY) {
            result = a * b;
        } else if (instruction.kind == InstructionKind::MAX) {
            result = max(a, b);
        } else if (instruction.kind == InstructionKind::MIN) {
            result = min(a, b);
        } else if (instruction.kind == InstructionKind::EQUAL) {
            result = a == b;
        } 
        stack[stack.length] = result;
        stack.length++;
    }
}

template <unsigned int UnitSize, unsigned int MaxInstructions>
/**
* @brief Kernel to run the programs.
*
* MaxConstants: maximum number of constants possible.
* NumInputs: number of input channels.
* MaxVariables: maximum number of variables.
* NumOutputs: number of output channels.
* UnitSize: dimensions of the area of responsibility of each thread. We
*   need to run each program on a single block, so each thread may need to
*   handle more than one cell.
* MaxInstructions: maximum number of instructions for each program.
*
* @param instructionBlock the block of instructions to use.
* @param constantBlock the block of constants to use.
* @param _inputs tensor of inputs. Shape: [#Programs, #Inputs, Y, X].
* @param _variables tensor of variables. Shape: [#Programs, #MaxVars, Y, X]
* @param _outputs tensor of outputs. Shape: [#Programs, #Outputs, Y, X].
* @param numInputs number of input channels.
* @param numVariables number of variable channels.
* @param numOutputs number of output channels.
*/
__global__ void runProgramKernel(
        const Array<const Program> programs,
        const Instruction* instructionBlock,
        const float* constantBlock,
        const float* inputs,
        float* _variables,
        float* _outputs,
        unsigned int numInputs, 
        unsigned int numVariables, 
        unsigned int numOutputs) {
    
    unsigned int programIndex = blockIdx.x;
    const Program& program = programs.data[programIndex];
    Array<const Instruction> instructions = program.getInstructions(instructionBlock);
    Array<const float> constants = program.getConstants(constantBlock);

    /// ASSUMPTION: unitSize * axisThreads = GridSize (GridSize evenly divides).
    unsigned int gridSizeX = UnitSize * blockDim.x;
    unsigned int gridSizeY = UnitSize * blockDim.x;
    unsigned int firstX = UnitSize * threadIdx.x;
    unsigned int firstY = UnitSize * threadIdx.y;

    // Slice tensors for convenience, factoring in the program index.
    float* variables = _variables + programIndex * (gridSizeX * gridSizeY * numVariables);
    float* outputs = _outputs + programIndex * (gridSizeX * gridSizeY * numOutputs);

    // Stacks of temporary values loaded or produced while running, one for
    // each sub-unit.
    float stackData[UnitSize * UnitSize * MaxInstructions];
    Array<float> stack[UnitSize][UnitSize];
    for (unsigned int ux = 0; ux < UnitSize; ux++) {
        for (unsigned int uy = 0; uy < UnitSize; uy++) {
            unsigned int offset = MaxInstructions * (uy * UnitSize + ux);
            stack[uy][ux].data = stackData + offset;
        }
    }

    for (unsigned int instructionIndex = 0; 
            instructionIndex < instructions.length; instructionIndex++) {
        const Instruction& instruction = instructions[instructionIndex];

        // We synchronize the block before each LOAD_VARIABLE to ensure that
        // all threads have finished writing any intermediate results to be 
        // read.
        bool needsSync = instruction.kind == InstructionKind::LOAD_VARIABLE;
        if (needsSync) {
            __syncthreads();
        }

        for (unsigned int ux = 0; ux < UnitSize; ux++) {
            for (unsigned int uy = 0; uy < UnitSize; uy++) {
                auto x = firstX + ux;
                auto y = firstY + uy;
                runInstruction(instruction, x, y, gridSizeX, gridSizeY, 
                        constants.data, inputs, variables, outputs, 
                        stack[uy][ux]);
            }
        }
    }
}


template <unsigned int UnitSize, unsigned int MaxInstructions>
/**
* @brief Runs programs on the device.
* 
* UnitSize: How many cells for each thread to manage. Each cell in the 
*   grid must be managed by a thread. Due to technical limitations, there 
*   can be at most 1024 device threads per program. Thus UnitSize must be
*   set such that (Y / UnitSize) * (X / UnitSize) <= 1024.
*   Note: For now, X and Y must divide evenly by UnitSize.
* MaxInstructions: Maximum number of instructions to run.
* 
* Regarding parameter shapes:
*   P: maximum number of programs.
*   N: maximum number of instructions.
*   C: number of constants available for programs to use.
*   I, V, O: number of input, variable or output channels respectively.
*   V: number of variable channels available for programs to use.
*   Y, X: dimensions of the grid.
* 
* @param programs buffer containing array of programs to be run, shape: [P].
* @param instructions block of instructions used by the programs, shape: [P, N].
* @param constants block of constants used by the programs, shape: [P, C].
* @param inputs tensor of input values, shape: [I, Y, X]. Each program is
*   provided the same input.
* @param variables tensor of variable values, shape: [P, V, Y, X].
* @param outputs tensor of output values, shape: [P, O, Y, X].
* @param numPrograms number of programs to run, in case it is desired to run
*   fewer programs than the buffer allows.
* @param autoTransfer for convenience, whether to automatically
*   copy from host->device programs, instructions, constants and inputs
*   before running the programs. And after, whether to automatically copy 
*   outputs from device->host. Variables are not automatically transferred.
*/
void runProgramsOnDevice(
        CrossBuffer<Program>& programs, 
        CrossBuffer<Instruction>& instructions,
        CrossBuffer<float>& constants,
        CrossBuffer<float>& inputs,
        CrossBuffer<float>& variables,
        CrossBuffer<float>& outputs,
        std::optional<unsigned int> _numPrograms = std::nullopt,
        bool autoTransfer = false) {       
    unsigned int numPrograms = _numPrograms.value_or(programs.dims[0]);

    // Assert all buffers contain enough room for the number of programs.
    assert(numPrograms <= programs.dims[0]);
    assert(numPrograms <= instructions.dims[0]);
    assert(numPrograms <= constants.dims[0]);
    assert(numPrograms <= variables.dims[0]);
    assert(numPrograms <= outputs.dims[0]);

    // Assert instructions holds at least MaxInstructions worth of room.
    assert(MaxInstructions <= instructions.dims[1]);

    // Assert grid tensors have same physical dimensions.
    assert(inputs.dims[1] == variables.dims[2] && inputs.dims[2] == variables.dims[3]);
    assert(inputs.dims[1] == outputs.dims[2] && inputs.dims[2] == outputs.dims[3]);

    // Assert UnitSize is set correctly.
    assert(inputs.dims[1] % UnitSize == 0);
    assert(inputs.dims[1] * inputs.dims[2] / UnitSize / UnitSize < 1024);

    // Transfer host to device if requested.
    if (autoTransfer) {
        programs.transferHostToDevice();
        instructions.transferHostToDevice();
        constants.transferHostToDevice();
        inputs.transferHostToDevice();
    }

    Array<const Program> programsDevice = programs.toDeviceArrayConst();
    programsDevice.length = numPrograms;


    auto gridSizeY = inputs.dims[1];
    auto gridSizeX = inputs.dims[2];
    auto kernelGridSize = dim3(numPrograms);
    auto kernelBlockSize = dim3(gridSizeX / UnitSize, gridSizeY / UnitSize);
    runProgramKernel<UnitSize, MaxInstructions>
        <<<kernelGridSize, kernelBlockSize>>>(programsDevice, 
        instructions.deviceBuffer, constants.deviceBuffer, 
        inputs.deviceBuffer, variables.deviceBuffer, outputs.deviceBuffer, 
        inputs.dims[0], variables.dims[1], outputs.dims[1]);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "runProgramKernel launch failed: %s\n", 
                cudaGetErrorString(cudaStatus));
        exit(-1);
    }

    // Transfer results device to host if requested.
    if (autoTransfer) {
        outputs.transferDeviceToHost();
    }
}