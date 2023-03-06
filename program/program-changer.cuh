#pragma once
#include "program.cuh"
#include <vector>
#include <algorithm>

/**
* @brief Program put into a format for easy mutation and change.
*/
struct MutableProgram {
    Program program;
    std::vector<Instruction> instructions;
    std::vector<float> constants;
};

/**
 * @brief Utility for creating, mutating and combining programs together.
 */
class ProgramChanger {

    /**
    * @brief Engine for generating random numbers.
    */
    std::default_random_engine r;

public:
    /**
    * @brief Constructs the ProgramChanger module.
    */
    ProgramChanger() {
        
    }

    /**
    * @brief Mutate the Program in place, performing one random modification.
    * 
    * @param program the program to mutate. After changing, this may have
    *    problems; use fixUp(...) before running the program.
    */
    void mutate(MutableProgram& program) {

        unsigned int operation = uniformInt(0, 6, r);

        // Change a constant value.
        if (operation == 0 && !program.constants.empty()) {
            unsigned int index = uniformInt<unsigned int>(0, program.constants.size(), r);
            
            // With high probability, change it by a small amount. With low 
            // probability, overwrite it entirely.
            float p = uniformReal<float>(r);
            if (p < 0.75f) {
                program.constants[index] += normalReal<float>(0.1f, r);
            } else if (p < 0.9f) {
                program.constants[index] += normalReal<float>(1.0f, r);
            } else {

                // Use of uniform gives an opportunity to introduce whole
                // number constants. Subsequent mutations can produce fractional
                // values again.
                program.constants[index] = (float)uniformInt(-10, 10, r);
            }

        // Add a new constant.
        } else if (operation == 1) {

            // Subsequent mutations can change this value.
            float value;
            if (uniformReal<float>(r) < 0.5) {
                value = 0;
            } else {
                value = 1;
            }
            program.constants.push_back(value);

        // Remove a constant.
        } else if (operation == 2 && !program.constants.size() > 2) {
            unsigned int loc = uniformInt<unsigned int>(0, 
                    program.constants.size() - 1, r);
            auto deleteSpot = program.constants.begin() + loc;
            program.constants.erase(deleteSpot);

        // Change number of variables this program can use.
        } else if (operation == 3) {

            // Due to unsigned logic, numVariables could become huge, but
            // fixUp(...) will take care of that later.
            program.program.numVariables += uniformInt<int>(0, 1, r) * 2 - 1;

        // Insert a new instruction.
        } if (operation == 3) {

            InstructionKind kind = randomInstructionKind();
            unsigned short argument = uniformInt<unsigned short>(0, 10000, r);

            unsigned int loc = uniformInt<unsigned int>(0, 
                    program.instructions.size(), r);
            auto insertSpot = program.instructions.begin() + loc;
            program.instructions.insert(insertSpot, {kind, argument});

        // Delete an instruction.
        } else if (operation == 4 && !program.instructions.empty()) {
            unsigned int loc = uniformInt<unsigned int>(0, 
                    program.instructions.size() - 1, r);
            auto deleteSpot = program.instructions.begin() + loc;
            program.instructions.erase(deleteSpot);

        // Modify an instruction.
        } else if (operation == 5 && !program.instructions.empty()) {
            unsigned int index = uniformInt<unsigned int>(0, program.instructions.size(), r);

            if (uniformReal<float>(r) < 0.5f) {
                program.instructions[index].kind = randomInstructionKind();
            } else {

                // Argument could wrap around but fixUp(...) can handle that.
                program.instructions[index].argument -= uniformInt<int>(-4, 4, r);
            }
        } 

        // Since we may have changed the lengths of these vectors, we must 
        // update the corresponding lengths in the Porgram.
        program.program.instructions.length = program.instructions.size();
        program.program.constants.length = program.constants.size();
    }

    /**
    * @brief Combines two programs together.
    * 
    * Note that this function respects the array offsets from each program. If
    * applicable, the instructions and constants pointers should point to the
    * start of the larger blocks that these programs store their data in.
    * 
    * @param a first program.
    * @param b second program.
    * @return combined program. May have problems; use fixUp(...) before 
    *   running the program.
    */
    MutableProgram combine(const MutableProgram& a, const MutableProgram& b) {

        // Randomly select instructions (in order) from the two programs.
        std::vector<Instruction> combinedInstructions;
        unsigned int maxInstructions = std::max(a.instructions.size(),
                b.instructions.size());
        for (unsigned int i = 0; i < maxInstructions; i++) {
            if (i < a.instructions.size() && uniformReal<float>(r) < 0.5f) {
                combinedInstructions.push_back(a.instructions[i]);
            }
            if (i < b.instructions.size() && uniformReal<float>(r) < 0.5) {
                combinedInstructions.push_back(b.instructions[i]);
            }
        }

        // Randomly select constants from the two programs.
        std::vector<float> combinedConstants;
        unsigned int maxConstants = std::max(
                a.constants.size(),
                b.constants.size());
        for (unsigned int i = 0; i < maxConstants; i++) {
            if (i < a.constants.size() && uniformReal<float>(r) < 0.5f) {
                combinedConstants.push_back(a.constants[i]);
            }
            if (i < b.constants.size() && uniformReal<float>(r) < 0.5f) {
                combinedConstants.push_back(b.constants[i]);
            }
        }

        // The combined variable count is between that of the two programs.
        auto minVar = std::min(a.program.getNumVariables(), b.program.getNumVariables());
        auto maxVar = std::max(a.program.getNumVariables(), b.program.getNumVariables());
        unsigned int combinedVarCount = uniformInt(minVar, maxVar, r);

        Program program(
                {0, combinedInstructions.size()},
                {0, combinedConstants.size()},
                combinedVarCount);
        return {program, combinedInstructions, combinedConstants};
    }

    /**
    * @brief Fixes up a program that may have become invalid after combination
    * or mutation, enforcing constraints.
    * 
    * @param program the program to fix up.
    * @param maxInstructions maximum number of instructions allowed.
    * @param maxConstants maximum number of constants allowed.
    * @param numInputs number of input channels.
    * @param maxVariables maximum number of variables allowed.
    * @param numOutputs number of output channels.
    */
    void fixUp(MutableProgram& program, 
            unsigned int maxInstructions,
            unsigned int maxConstants,
            unsigned int numInputs,
            unsigned int maxVariables,
            unsigned int numOutputs) {

        // Enforce program limits.
        if (program.instructions.size() > maxInstructions) {
            program.instructions.erase(program.instructions.begin() + maxInstructions, 
                    program.instructions.end());
        }
        if (program.constants.size() > maxConstants) {
            program.constants.erase(program.constants.begin() + maxConstants, 
                    program.constants.end());
        } else if (program.constants.empty()) {
            program.constants.push_back(0);
        }
        program.program.numVariables = std::min<unsigned int>(
                std::max<unsigned int>(program.program.numVariables, 1), 
                maxVariables - 1
        );

        // Update program's length fields to match vector sizes.
        program.program.instructions.length = program.instructions.size();
        program.program.constants.length = program.constants.size();

        // Enforce instruction argument limits.
        for (Instruction& instruction : program.instructions) {

            // Remember, each input or variable LOAD can take from neighbours,
            // thus there are 9 positions for each channel.
            if (instruction.kind == InstructionKind::LOAD_CONSTANT) {
                instruction.argument = instruction.argument % program.constants.size();
            } else if (instruction.kind == InstructionKind::LOAD_INPUT) {
                instruction.argument = instruction.argument % (9 * numInputs);
            } else if (instruction.kind == InstructionKind::LOAD_VARIABLE) {
                instruction.argument = instruction.argument % (9 * program.program.getNumVariables());
            } else if (instruction.kind == InstructionKind::STORE_VARIABLE) {
                instruction.argument = instruction.argument % program.program.getNumVariables();
            } else if (instruction.kind == InstructionKind::STORE_OUTPUT) {
                instruction.argument = instruction.argument % numOutputs;
            }
        }
    }

private:

    /**
    * @brief Generates a random instruction kind.
    * 
    * @return any possible valid instruction kind.
    */
    InstructionKind randomInstructionKind() {
        return InstructionKind::Enum(uniformInt<unsigned short>(
                InstructionKind::FIRST, 
                InstructionKind::LAST, 
                r
        ));
    }

};