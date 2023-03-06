#pragma once
#include "util/util.cuh"
#include "program/program-changer.cuh"
#include "jsoncpp.h"
#include <vector>
#include <assert.h>
#include <optional>
#include <functional>
#include <sstream>

/**
* @brief Environment where programs can evolve.
*
* TODO: This is just a basic environment. In future, we can emulate having
* separate biomes and ecological niches. Moreover, we can introduce measures to
* prevent the survival list from just being filled with near-identical copies of 
* the same program.
*/
class EvolutionEnvironment {
public:

    /**
    * @brief Format of the scoring hook.
    * 
    * programs: an array of programs to be evaluated.
    * scores: an array of program scores from evaluating the programs.
    */
    using ScoreHook = std::function<void(Array<const MutableProgram>, float*)>;

    /**
    * @brief Describes the problem to solve. 
    */
    struct ProblemInfo {
        ScoreHook hook;
        unsigned int maxInstructions;
        unsigned int maxConstants;
        unsigned int numInputs;
        unsigned int maxVariables;
        unsigned int numOutputs;
    };

private:

    // Associates a program with its score.
    using ScoredProgram = std::pair<MutableProgram, float>;

    /**
    * @brief Place to store programs to be processed and scored.
    */
    std::vector<MutableProgram> intakePrograms;

    /**
    * @brief Set of current surviving programs. 
    * 
    * This will become temporarily enlarged while processing intake programs.
    */
    std::vector<ScoredProgram> survivalSet;

    /**
    * @brief Information about the problem to solve.
    */
    ProblemInfo problemInfo;

    /**
    * @brief Size of the set of surviving programs.
    * 
    * Only the top scoring programs survive after each iteration. 
    * 
    * Invariant: before and after iterate(), the size of survivalSet must be
    * no greater than this number.
    */
    unsigned int surviveSetSize;

    /**
    * @brief Number of programs that will be in the intake set during
    *   evaluation.
    * 
    * When generating new programs, exactly enough programs to fill the intake
    * set will be generated.
    */
    unsigned int intakeSize;

    /**
    * @brief Tracks the current iteration/generation number.
    */
    unsigned int iteration;

    /**
    * @brief Random engine for generating programs. 
    */
    std::default_random_engine r;

    /**
    * @brief Helper module for changing programs.
    */
    ProgramChanger changer;

public:

    /**
    * @brief Constructs the environment.
    * 
    * @param scoreHook runs candidate programs on the problem to optimize and
    *   assigns a score for each one.
    * @param surviveSetSize size of the set of surviving programs. Only the
    *   top scoring programs survive after each iteration. Programs that score
    *   less than this number of top solutions are deleted.
    * @param intakeSize number of programs in the intake set during evaluation.
    *   This environment will always generate exactly enough programs to fill 
    *   this set.
    */
    EvolutionEnvironment(const ProblemInfo& problemInfo,
            unsigned int surviveSetSize,
            unsigned int intakeSize) : problemInfo(problemInfo),
            surviveSetSize(surviveSetSize),
            intakeSize(intakeSize), iteration(0) {

    }

    /**
    * @brief Runs a single iteration of the evolution process.
    */
    void iterate() {

        // Combine and mutate existing programs to create more.
        generatePrograms();

        // This temporarily enlarges survivalSet. We must cull the lowest
        // scoring programs to restore the invariant.
        processIntakeSet();

        // Sort and cull programs; highest score goes first.
        std::sort(survivalSet.begin(), survivalSet.end(), 
            [](const ScoredProgram& a, const ScoredProgram& b) {
                return a.second > b.second;
            }
        );
        survivalSet.erase(survivalSet.begin() + 
                std::min<size_t>(surviveSetSize, survivalSet.size()), 
                survivalSet.end());
        assert(survivalSet.size() <= surviveSetSize);

        if (iteration % 100 == 0) {
            printf("Iteration: %d; Best Score: %f\n", iteration, 
                    survivalSet[0].second);
        }
        if (iteration % 1000 == 999) {
            printf("%s\n", exportJSON(1).c_str());
        }

        iteration++;
    }

private:

    /**
    * @brief Export top-n programs into a different format.
    * 
    * @param topN number of programs to export.
    */
    std::string exportJSON(unsigned int topN) const {

        using json = nlohmann::json;
        using namespace nlohmann::literals;

        json::array_t programJSONs;
        for (unsigned int i = 0; i < std::min<unsigned int>(topN, survivalSet.size()); i++) {
            const MutableProgram& program = survivalSet[0].first;

            json::array_t instructions;
            for (const Instruction& instruction : program.instructions) {
                auto kindString = InstructionKind::DISPLAY[instruction.kind.value];
                instructions.push_back({
                    {"name", kindString},
                    {"argument", instruction.argument}
                });
            }

            json programJSON = {
                {"score", survivalSet[0].second},
                {"instructions", instructions},
                {"constants", program.constants}
            };
            programJSONs.push_back(programJSON);
        }
        return ((json)programJSONs).dump();
    }

    /**
    * @brief Generates new programs into the intake set. 
    */
    void generatePrograms() {
        while (intakePrograms.size() < intakeSize) {

            // Generate new programs if there's few existing ones to combine OR
            // with a small probability.
            std::optional<MutableProgram> _newProgram;
            if (uniformReal<float>(r) < 0.05f || (intakePrograms.size() < 50 && 
                    survivalSet.empty())) {
                _newProgram.emplace(MutableProgram());

            // Otherwise combine existing programs together 
            } else {

                const MutableProgram& a = sampleParent();
                const MutableProgram& b = sampleParent();
                _newProgram.emplace(changer.combine(a, b));
            }
            MutableProgram newProgram = _newProgram.value();

            // Mutate the program.
            unsigned int numMutations;
            if (iteration == 0) {
                numMutations = uniformInt<unsigned int>(1, 200, r);
            } else {
                numMutations = uniformInt<unsigned int>(1, 50, r);
            }
            for (int i = 0; i < numMutations; i++) {
                changer.mutate(newProgram);
            }
            changer.fixUp(newProgram, 
                    problemInfo.maxInstructions, 
                    problemInfo.maxConstants, 
                    problemInfo.numInputs,
                    problemInfo.maxVariables,
                    problemInfo.numOutputs);
            intakePrograms.push_back(newProgram);
        }
    }

    /**
    * @brief Processes the set of intake programs.
    * 
    * These are programs that haven't been assigned a score yet. This will run
    * them together, assign a score and add them to the member list.
    */
    void processIntakeSet() {

        // Run the scoreHook. We pass it an array that it should use to set
        // the scores in.
        float* scores = new float[intakePrograms.size()];
        Array<const MutableProgram> programs(intakePrograms.data(),
                intakePrograms.size());
        problemInfo.hook(programs, scores);

        for (unsigned int i = 0; i < intakePrograms.size(); i++) {
            survivalSet.push_back({intakePrograms[i], scores[i]});
        }
        delete[] scores;
        intakePrograms.clear();
    }

    /**
    * @brief Obtains an existing program to be used as a parent. 
    *
    * @return constant-reference to the parent program to use.
    */
    const MutableProgram& sampleParent() {
        
        // We prefer survivalSet but can use intakePrograms.
        if (survivalSet.empty()) {
            size_t index = uniformInt<size_t>(0, intakePrograms.size(), r);
            return intakePrograms[index];
        } else {
            size_t index = uniformInt<size_t>(0, survivalSet.size(), r);
            return survivalSet[index].first;
        }
    }

};