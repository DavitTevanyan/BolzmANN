#include "TrainingData.h"
#include <sstream>

using namespace ANN;

namespace ANN {

std::vector<Sample> getTrainSet(const std::string& fileName)
{
    TrainingData samples(fileName);

    std::vector<Sample> trainSet;
    while (!samples.allRead())
    {
        auto input  = samples.getValues();
        auto target = samples.getValues();
        trainSet.push_back({ input, target });
    }
    return trainSet;
}

TrainingData::TrainingData(const std::string& fileName)
    : dataFile_(fileName.c_str())
{
    if (!dataFile_)
        throw std::runtime_error("ERROR: Could not open training data file: " + fileName);
}

std::vector<double> TrainingData::getValues()
{
    std::vector<double> input;

    std::string line;
    getline(dataFile_, line);
    std::stringstream ss(line);

    double x;
    while (ss >> x) 
        input.emplace_back(x);

    return input;
}

///////////////////////////////////////////////////////
// Samples for profiling (to avoid slow file access) //
///////////////////////////////////////////////////////

Sample a1 = { { 1.0, 0.0 }, { 0.0 } };
Sample a2 = { { 1.0, 1.0 }, { 1.0 } };
Sample a3 = { { 0.0, 1.0 }, { 0.0 } };
Sample a4 = { { 0.0, 0.0 }, { 0.0 } };

std::vector<Sample> AND = { a1, a2, a3, a4 };

Sample x1 = { { 1.0, 0.0 }, { 1.0 } };
Sample x2 = { { 1.0, 1.0 }, { 0.0 } };
Sample x3 = { { 0.0, 1.0 }, { 1.0 } };
Sample x4 = { { 0.0, 0.0 }, { 0.0 } };

std::vector<Sample> XOR = { x1, x2, x3, x4 };

} // namespace ANN