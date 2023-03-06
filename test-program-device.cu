#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "program/device-runner.cuh"
#include "program/program-changer.cuh"
#include "program/program.cuh"
#include "util/cross-buffer.cuh"
#include "util/util.cuh"

/**
* Saves repetitive syntax for running the programs on the device. Just need
* to make sure all variables are defined.
*/
#define RUN_PROGRAMS(NumPrograms) \
    runProgramsOnDevice<UNIT_SIZE, MAX_INSTRUCTIONS>( \
    programs, instructionBlock, constantBlock, \
    inputs, variables, outputs, \
    NumPrograms, true);

/**
* Helper to write program data into the buffers. Just need to make sure 
* all variables are defined.
*
* i: index of the program to assign.
*/
#define WRITE_PROGRAM(i) \
    { \
        auto programIndex = i; \
        programs.get(programIndex) = Program( \
            {programIndex * MAX_INSTRUCTIONS, instructions.size()}, \
            {programIndex * MAX_CONSTANTS, constants.size()}, \
            numVariables); \
        instructionBlock.copyIn(Array<Instruction>(instructions.data(), \
                instructions.size()), programIndex, 0); \
        constantBlock.copyIn(Array<float>(constants.data(), constants.size()), \
                programIndex, 0); \
    }


TEST_CASE("Test tiny grid on the device.") {
    const unsigned int UNIT_SIZE = 1;
    const unsigned int MAX_INSTRUCTIONS = 64;
    const unsigned int MAX_CONSTANTS = 16;
    const unsigned int MAX_PROGRAMS = 2;

    CrossBuffer<Program> programs(NULL, NULL, MAX_PROGRAMS);
    CrossBuffer<Instruction> instructionBlock(NULL, NULL, MAX_PROGRAMS, MAX_INSTRUCTIONS);
    CrossBuffer<float> constantBlock(NULL, NULL, MAX_PROGRAMS, MAX_CONSTANTS);
    CrossBuffer<float> inputs(NULL, NULL, 4, 1, 1);
    CrossBuffer<float> variables(NULL, NULL, MAX_PROGRAMS, 4, 1, 1);
    CrossBuffer<float> outputs(NULL, NULL, MAX_PROGRAMS, 4, 1, 1);

    inputs.get(0, 0, 0) = -6;
    inputs.get(1, 0, 0) = 3;
    inputs.get(2, 0, 0) = 2;
    inputs.get(3, 0, 0) = 0.125f;

    SUBCASE("Basic load-store") {
        std::vector<Instruction> instructions{
            {InstructionKind::LOAD_CONSTANT, 1},
            {InstructionKind::STORE_OUTPUT, 0},
        };
        std::vector<float> constants{2, 4};
        unsigned int numVariables = 0;
        WRITE_PROGRAM(0);

        RUN_PROGRAMS(1);
        CHECK(outputs.get(0, 0, 0, 0) == 4);
    }
    SUBCASE("Basic multiply") {
        std::vector<Instruction> instructions{
            {InstructionKind::LOAD_CONSTANT, 2},
            {InstructionKind::LOAD_INPUT, 4},
            {InstructionKind::MULTIPLY, 0},
            {InstructionKind::STORE_OUTPUT, 0},
        };
        std::vector<float> constants{-0.5f, 2, 4};
        unsigned int numVariables = 0;
        WRITE_PROGRAM(0);

        RUN_PROGRAMS(1);
        CHECK(outputs.get(0, 0, 0, 0) == -24);
    }
    SUBCASE("Two simple programs") {
        {
            std::vector<Instruction> instructions{
                {InstructionKind::LOAD_INPUT, 3 * 9 + 4},
                {InstructionKind::STORE_OUTPUT, 0},
            };
            std::vector<float> constants{2, 3};
            unsigned int numVariables = 0;
            WRITE_PROGRAM(0);
        }

        {
            std::vector<Instruction> instructions{
                {InstructionKind::LOAD_INPUT, 2 * 9 + 4},
                {InstructionKind::LOAD_CONSTANT, 0},
                {InstructionKind::STORE_OUTPUT, 0},
                {InstructionKind::STORE_OUTPUT, 1},
            };
            std::vector<float> constants{-0.5f, 0.75f};
            unsigned int numVariables = 0;
            WRITE_PROGRAM(1);
        }

        RUN_PROGRAMS(2);

        // First program.
        CHECK(outputs.get(0, 0, 0, 0) == 0.125f);

        // Second program.
        CHECK(outputs.get(1, 0, 0, 0) == -0.5f);
        CHECK(outputs.get(1, 1, 0, 0) == 2);
    }
    SUBCASE("Two simple programs that use variable channels.") {
        {
            std::vector<Instruction> instructions{
                {InstructionKind::LOAD_INPUT, 1 * 9 + 4},
                {InstructionKind::STORE_VARIABLE, 2},
                {InstructionKind::LOAD_CONSTANT, 0},
                {InstructionKind::LOAD_VARIABLE, 2 * 9 + 4},
                {InstructionKind::MULTIPLY, 0},
                {InstructionKind::STORE_OUTPUT, 0},
                {InstructionKind::LOAD_VARIABLE, 2 * 9 + 4},
                {InstructionKind::GT_0, 0},
                {InstructionKind::STORE_OUTPUT, 1},
            };
            std::vector<float> constants{5};
            unsigned int numVariables = 4;
            WRITE_PROGRAM(0);
        }

        {
             std::vector<Instruction> instructions{
                {InstructionKind::LOAD_INPUT, 2 * 9 + 4},
                {InstructionKind::STORE_VARIABLE, 2},
                {InstructionKind::LOAD_CONSTANT, 0},
                {InstructionKind::LOAD_VARIABLE, 2 * 9 + 4},
                {InstructionKind::MULTIPLY, 0},
                {InstructionKind::STORE_OUTPUT, 0},
                {InstructionKind::LOAD_CONSTANT, 1},
                {InstructionKind::STORE_OUTPUT, 1},
            };
            std::vector<float> constants{5, -16};
            unsigned int numVariables = 0;
            WRITE_PROGRAM(1);
        }

        RUN_PROGRAMS(2);

        // First program.
        CHECK(outputs.get(0, 0, 0, 0) == 15);
        CHECK(outputs.get(0, 1, 0, 0) == 1);

        // Second program.
        CHECK(outputs.get(1, 0, 0, 0) == 10);
        CHECK(outputs.get(1, 1, 0, 0) == -16);
    }
}

TEST_CASE("Test larger grid.") {
    const unsigned int UNIT_SIZE = 1;
    const unsigned int MAX_INSTRUCTIONS = 64;
    const unsigned int MAX_CONSTANTS = 16;
    const unsigned int MAX_PROGRAMS = 2;

    CrossBuffer<Program> programs(NULL, NULL, MAX_PROGRAMS);
    CrossBuffer<Instruction> instructionBlock(NULL, NULL, MAX_PROGRAMS, MAX_INSTRUCTIONS);
    CrossBuffer<float> constantBlock(NULL, NULL, MAX_PROGRAMS, MAX_CONSTANTS);
    CrossBuffer<float> inputs(NULL, NULL, 4, 3, 3);
    CrossBuffer<float> variables(NULL, NULL, MAX_PROGRAMS, 4, 3, 3);
    CrossBuffer<float> outputs(NULL, NULL, MAX_PROGRAMS, 4, 3, 3);

    SUBCASE("Test Conway's Game of Life") {
        inputs.get(0, 0, 1) = 1;
        inputs.get(0, 1, 1) = 1;
        inputs.get(0, 2, 1) = 1;

        std::vector<Instruction> instructions{

            // Count alive neighbours.
            {InstructionKind::LOAD_INPUT, 0},
            {InstructionKind::LOAD_INPUT, 1},
            {InstructionKind::LOAD_INPUT, 2},
            {InstructionKind::LOAD_INPUT, 3},
            {InstructionKind::LOAD_INPUT, 5},
            {InstructionKind::LOAD_INPUT, 6},
            {InstructionKind::LOAD_INPUT, 7},
            {InstructionKind::LOAD_INPUT, 8},
            {InstructionKind::ADD, 0},
            {InstructionKind::ADD, 0},
            {InstructionKind::ADD, 0},
            {InstructionKind::ADD, 0},
            {InstructionKind::ADD, 0},
            {InstructionKind::ADD, 0},
            {InstructionKind::ADD, 0},
            {InstructionKind::DUPLICATE, 0},

            // Compute + Store N == 3
            {InstructionKind::LOAD_CONSTANT, 0},
            {InstructionKind::EQUAL, 0},
            {InstructionKind::STORE_VARIABLE, 0},

            // Compute N == 2 & this_alive
            {InstructionKind::LOAD_CONSTANT, 1},
            {InstructionKind::EQUAL, 0},
            {InstructionKind::LOAD_INPUT, 4},
            {InstructionKind::MULTIPLY, 0},

            // OR the two results above.
            {InstructionKind::LOAD_VARIABLE, 4},
            {InstructionKind::ADD, 0},
            {InstructionKind::GT_0, 0},
            {InstructionKind::STORE_OUTPUT, 0}
        };
        std::vector<float> constants{3, 2};
        unsigned int numVariables = 1;
        WRITE_PROGRAM(0);
        RUN_PROGRAMS(1);

        CHECK(outputs.get(0, 0, 0, 0) == 0);
        CHECK(outputs.get(0, 0, 0, 1) == 0);
        CHECK(outputs.get(0, 0, 0, 2) == 0);
        CHECK(outputs.get(0, 0, 1, 0) == 1);
        CHECK(outputs.get(0, 0, 1, 1) == 1);
        CHECK(outputs.get(0, 0, 1, 2) == 1);
        CHECK(outputs.get(0, 0, 2, 0) == 0);
        CHECK(outputs.get(0, 0, 2, 1) == 0);
        CHECK(outputs.get(0, 0, 2, 2) == 0);
    }
}