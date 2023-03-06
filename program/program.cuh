#pragma once
#include <util/util.cuh>

struct InstructionKind;
struct Instruction;

/**
* @brief An individual program.
*
* This is expressed in a form that is both amenable to evolution and useable
* in device code.
*/
class Program {
friend class ProgramChanger;
private:

    /// TODO: Refactor. It would be better to split this into a host-side 
    /// evolution-friendly version and a paired-down device-friendly version. 
    /// The original concept used tightly packed constant and instructions 
    /// spaces shared between multiple programs; but now instead, each program
    /// has its own predictable, aligned and dedicated space for these.

    /**
    * @brief The list of instructions to run.
    * 
    * Use of offset format eliminates the need for pointer fix-up when copying
    * the program to device.
    */
    ArrayByOffset<Instruction> instructions;

    /**
    * @brief The list of constants the program can use.
    */
    ArrayByOffset<float> constants;

    /**
    * Total number of variables the program can use.
    * 
    * Using a configurable value instead of defaulting to using a large one
    * should help focus evolution, allowing faster convergence.
    */
   unsigned int numVariables;

public:

    /**
    * @brief Default-constructs this program.
    */
    Program() : instructions{0, 0}, constants{0, 0}, numVariables(0) {

    }

    /**
    * @brief Constructs the Program.
    * 
    * @param instructions the array of instructions that make up the program.
    * @param constants the array of constants the program can use.
    * @param numVariables number of variables the program can use.
    */
    Program(ArrayByOffset<Instruction> instructions, 
            ArrayByOffset<float> constants, unsigned int numVariables) : 
            instructions(instructions), constants(constants),
            numVariables(numVariables) {

    }

    /**
    * @brief Gets the number of variables this program can use.
    * 
    * In a properly formed program, any instructions to load or store from a
    * variable must be restricted to use only variables with indices within 
    * this limit.
    * 
    * @return number of variables that the program can use.
    */
    inline unsigned int getNumVariables() const {
        return numVariables;
    }

    /**
    * @brief Gets the array of instructions.
    * 
    * @param blockStart pointer to the start of the Instruction data block
    *   that this program uses a slice of.
    */
    inline __device__ __host__ Array<const Instruction> getInstructions(
            const Instruction* blockStart) const {
        return instructions.toArrayConst(blockStart);
    }

    /**
    * @brief Gets the array of constants.
    * 
    * @param blockStart pointer to the start of the float data block
    *   that this program uses a slice of to store constants.
    */
    inline __device__ __host__ Array<const float> getConstants(
            const float* blockStart) const {
        return constants.toArrayConst(blockStart);
    }

    /**
    * @brief Change the offsets in the instructions and constants array.
    * 
    * For use when a program is moved to work with different instruction and
    * constant blocks.
    */
    inline void changeOffsets(unsigned int instructionOffset,
            unsigned int constantOffset) {
        instructions.offset = instructionOffset;
        constants.offset = constantOffset;
    }

};

/**
* @brief Enumerates the types of instructions in the program.
*
* LOAD variants copy items from inputs, variables, or constants and pushes
*   them to the top of the stack.
* STORE variants pop items off the top of the stack and places them in 
*   in outputs or variables.
* ADD/SUB/etc. pop the top item(s) from the stack, run the operation
*   and push the result onto the top of the stack.
*
* Uses an enum-in-struct containment pattern to allow adding helper methods.
*/
struct InstructionKind {

    enum Enum : unsigned short {
        FIRST = 0,

        FIRST_LOAD = 0,
        LOAD_CONSTANT = 0,
        LOAD_INPUT = 1,
        LOAD_VARIABLE = 2,
        LAST_LOAD = 2,

        FIRST_STORE = 3,
        STORE_OUTPUT = 3,
        STORE_VARIABLE = 4,
        LAST_STORE = 4,

        FIRST_UNARY = 5,
        GT_0 = 5,
        RECIPRICOL = 6,
        ABS = 7,
        DUPLICATE = 8,
        LAST_UNARY = 8,

        FIRST_BINARY = 9,
        ADD = 9,
        SUB = 10,
        MULTIPLY = 11,
        MIN = 12,
        MAX = 13,
        EQUAL = 14,
        LAST_BINARY = 14,

        LAST = 14
    };

    static constexpr const char* DISPLAY[LAST + 1] = {
        "Load Constant",
        "Load Input",
        "Load Variable",
        "Store Output",
        "Store Variable",
        "X > 0",
        "1 / X",
        "|X|",
        "Duplicate",
        "+",
        "-",
        "*",
        "min",
        "max",
        "==" 
    };

    /**
    * @brief The internal value of the enum.
    */
    Enum value;

    /**
    * @brief Default constructor.
    */
    inline InstructionKind() : value(LOAD_INPUT) {

    }

    /**
    * @brief Constructs from Enum.
    * 
    * @param value the enum value this will use.
    */
    inline InstructionKind(Enum value) : value(value) {

    }

    /**
    * @brief Indicates if this is equal to the given InstructionKind enum.
    */
    bool __host__ __device__ operator==(Enum other) const {
        return value == other;
    }

    /**
    * @brief Indicates if this is not equal to the given InstructionKind enum.
    */
    bool __host__ __device__ operator!=(Enum other) const {
        return value != other;
    }

    /**
    * @brief Indicates if this InstructionKind is a LOAD.
    */
    bool __host__ __device__ isLoad() const {
        return value >= FIRST_LOAD && value <= LAST_LOAD;
    }

    /**
    * @brief Indicates if this InstructionKind is a STORE.
    */
    bool __host__ __device__ isStore() const {
        return value >= FIRST_STORE && value <= LAST_STORE;
    }

    /**
    * @brief Indicates if this InstructionKind is a unary operation.
    */
    bool __host__ __device__ isUnary() const {
        return value >= FIRST_UNARY && value <= LAST_UNARY;
    }

    /**
    * @brief Indicates if this InstructionKind is a binary operation.
    */
    bool __host__ __device__ isBinary() const {
        return value >= FIRST_BINARY && value <= LAST_BINARY;
    }
};

/**
* @brief An instruction in the program.
*/
struct Instruction {
    
    /**
    * @brief The kind of instruction.
    */
    InstructionKind kind;

    /**
    * @brief The argument of the instruction.
    * 
    * Not all instructions require an argument. For example, binary and 
    * unary ops just look in the value stack.
    */
    unsigned short argument;

};