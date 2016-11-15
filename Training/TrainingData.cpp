#include "TrainingData.h"
#include <sstream>

using namespace ANN;

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