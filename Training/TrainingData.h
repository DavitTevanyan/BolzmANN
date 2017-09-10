#pragma once 

#include <fstream>
#include <vector>
#include <string>

namespace ANN
{
    struct Sample
    {
        std::vector<double> input;
        std::vector<double> target;
    };

    std::vector<Sample> getTrainSet(const std::string& fileName);

    class TrainingData
    {
    public:
        explicit TrainingData(const std::string& filename);
        std::vector<double> getValues();
        bool allRead() { return dataFile_.eof(); }

    private:
        std::ifstream dataFile_;
    };

///////////////////////////////////////////////////////
// Samples for profiling (to avoid slow file access) //
///////////////////////////////////////////////////////

    extern std::vector<Sample> AND;
    extern std::vector<Sample> XOR;

} // namespace ANN