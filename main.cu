#include "evolution/environment.cuh"
#include "program/program-changer.cuh"
#include "program/device-runner.cuh"
#include "util/cross-buffer.cuh"
#include "util/util.cuh"
#include <stdio.h>

/// This is a demo application that demonstrates how to use this program 
/// evolution software.
///
/// The objective of this demo is to create a program that blurs the 4 cardinal
/// neighbours of each point:
///     Output = (Top + Bottom + Left + Right) / 4

const unsigned int NUM_TEST_CASES = 4;
const unsigned int GRID_SIZE = 4;
float TEST_INPUTS[NUM_TEST_CASES][GRID_SIZE][GRID_SIZE] = {
    {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    },
    {
        {1, 2, 3, 4},
        {1, 3, 2, 5},
        {0, 5, 0, 3},
        {2, 1, 4, 6}
    },
    {
        {0, -0.5f,  3, 3},
        {2,     3,  1, 4},
        {4, -1.5f,  4, 0.5f},
        {1,     2,  0, -0.125f}
    },
    {
        {4, 4, 4, 4},
        {4, 4, 4, 4},
        {4, 4, 4, 4},
        {4, 4, 4, 4},
    }
};
float TEST_OUTPUTS[NUM_TEST_CASES][GRID_SIZE][GRID_SIZE] = {
    {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    },
    {
        {0.75f, 1.75f, 2,     2},
        {1,     2.5f,  2.75f, 2.25f},
        {2,     1,     3.5f,  2.75f},
        {0.25f, 2.75f, 1.75f, 1.75f}
    },
    {
        {0.375f, 1.5f,    0.875f,   1.75f},
        {1.75f,  0.25f,   3.5f,     1.125f},
        {0.375f, 3.25f,   0.0f,      1.96875f},
        {1.5f,   -0.125f,  1.46875f, 0.125f}
    },
    {
        {2, 3, 3, 2},
        {3, 4, 4, 3},
        {3, 4, 4, 3},
        {2, 3, 3, 2}
    },
};

int main() {

    // Set up resources for running the scoring hook.
    /// TODO: These can be factored into a helper structure.
    const unsigned int UNIT_SIZE = 1;
    const unsigned int MAX_INSTRUCTIONS = 256;
    const unsigned int MAX_CONSTANTS = 16;
    const unsigned int MAX_VARIABLES = 16;
    const unsigned int MAX_PROGRAMS = 256;
    const unsigned int NUM_INPUTS = 1;
    const unsigned int NUM_OUTPUTS = 1;
    const unsigned int GRID_SIZE = 4;
    CrossBuffer<Program> programBuffer(NULL, NULL, MAX_PROGRAMS);
    CrossBuffer<Instruction> instructionBlock(NULL, NULL, MAX_PROGRAMS, MAX_INSTRUCTIONS);
    CrossBuffer<float> constantBlock(NULL, NULL, MAX_PROGRAMS, MAX_CONSTANTS);
    CrossBuffer<float> inputs(NULL, NULL, NUM_INPUTS, GRID_SIZE, GRID_SIZE);
    CrossBuffer<float> variables(NULL, NULL, MAX_PROGRAMS, MAX_VARIABLES, GRID_SIZE, GRID_SIZE);
    CrossBuffer<float> outputs(NULL, NULL, MAX_PROGRAMS, NUM_OUTPUTS, GRID_SIZE, GRID_SIZE);

    // Create the scoring hook to evaluate candidate programs.
    EvolutionEnvironment::ScoreHook scoreHook = 
            [&](Array<const MutableProgram> programs, float* outScores) 
    {
        assert(programs.length == MAX_PROGRAMS);

        // Set up the program-related buffers and initialize scores to 0.
        for (unsigned i = 0; i < programs.length; i++) {
            const auto& mProgram = programs[i];
            programBuffer.get(i) = mProgram.program;
            instructionBlock.copyIn(Array<const Instruction>(mProgram.instructions), i, 0);
            constantBlock.copyIn(Array<const float>(mProgram.constants), i, 0);

            programBuffer.get(i).changeOffsets(i * MAX_INSTRUCTIONS, i * MAX_CONSTANTS);
            outScores[i] = 0;
        }
        
        // Run the programs on the test cases.
        for (unsigned int c = 0; c < NUM_TEST_CASES; c++) {

            Array<float> caseInput((float*)TEST_INPUTS + c, GRID_SIZE * GRID_SIZE);
            inputs.copyIn(caseInput, 0, 0, 0);
            variables.clear(false, true);

            runProgramsOnDevice<UNIT_SIZE, MAX_INSTRUCTIONS>(programBuffer, 
                    instructionBlock, constantBlock, inputs, variables, 
                    outputs, std::nullopt, true);

            for (unsigned int programIndex = 0; programIndex < programs.length; 
                    programIndex++) {
                
                // L1 penalty between program's result and ground truth.
                float score = 0;
                for (unsigned int y = 0; y < GRID_SIZE; y++) {
                    for (unsigned int x = 0; x < GRID_SIZE; x++) {
                        float expected = TEST_OUTPUTS[c][y][x];
                        float actual = outputs.get(programIndex, 0, y, x);
                        score -= abs(expected - actual);
                    }
                }
                outScores[programIndex] += score;
            }
        }
    };

    EvolutionEnvironment::ProblemInfo info = {scoreHook, MAX_INSTRUCTIONS, 
            MAX_CONSTANTS, NUM_INPUTS, MAX_VARIABLES, NUM_OUTPUTS};
    EvolutionEnvironment environment(info, 256, MAX_PROGRAMS);
    for (unsigned int i = 0; i < 100000; i++) {
        environment.iterate();
    }

    return 0;
}