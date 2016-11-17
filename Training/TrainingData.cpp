#include "TrainingData.h"
#include <sstream>

using namespace ANN;

namespace ANN {

std::vector<Sample> getTrainSet()
{
    TrainingData samples("and.txt");
    std::vector<Sample> trainSet;
    while (!samples.allRead())
        trainSet.push_back({ samples.getNextInput(), samples.getNextTarget() });
    return trainSet;
}

TrainingData::TrainingData(const std::string& fileName)
    : trainingDataFile_(fileName.c_str())
{
    // nothing
}

std::vector<double> TrainingData::getNextInput()
{
    std::vector<double> input;

    std::string line;
    getline(trainingDataFile_, line);
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
    std::vector<double> targetOutputVals;

    std::string line;
    getline(trainingDataFile_, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;
    if (label == "out:")
    {
        double oneValue;
        while (ss >> oneValue) 
        {
            targetOutputVals.emplace_back(oneValue);
        }
    }

    return targetOutputVals;
}

// Samples for profiling (to avoid slow file access)
Sample s1 = { { 1.0, 0.0 }, { 0.0 } };
Sample s2 = { { 1.0, 1.0 }, { 1.0 } };
Sample s3 = { { 0.0, 1.0 }, { 0.0 } };
Sample s4 = { { 0.0, 0.0 }, { 0.0 } };

std::vector<Sample> AND = { s1, s2, s3, s4 };

} // namespace ANN