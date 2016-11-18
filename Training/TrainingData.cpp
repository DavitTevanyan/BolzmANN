#include "TrainingData.h"
#include <sstream>

using namespace ANN;

namespace ANN {

std::vector<Sample> getTrainSet(const std::string& fileName)
{
    TrainingData samples(fileName);
    std::vector<Sample> trainSet;
    while (!samples.allRead())
        trainSet.push_back({ samples.getNextInput(), samples.getNextTarget() });
    return trainSet;
}

TrainingData::TrainingData(const std::string& fileName)
    : dataFile_(fileName.c_str())
{
    // nothing
}

std::vector<double> TrainingData::getNextInput()
{
    std::vector<double> input;

    std::string line;
    getline(dataFile_, line);
    std::stringstream strmLine(line);

    std::string label;
    strmLine >> label;
    if (label == "in:")
    {
        double value;
        while (strmLine >> value) 
        {
            input.emplace_back(value);
        }
    }

    return input;
}

std::vector<double> TrainingData::getNextTarget()
{
    std::vector<double> target;

    std::string line;
    getline(dataFile_, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;
    if (label == "out:")
    {
        double oneValue;
        while (ss >> oneValue) 
        {
            target.emplace_back(oneValue);
        }
    }

    return target;
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