#include "TrainingData.h"
#include <sstream>

using namespace ANN;

namespace ANN {

std::vector<Data> getTrainSet()
{
    TrainingData data("trainingData.txt");
    std::vector<Data> trainSet;
    while (!data.isEof())
        trainSet.push_back({ data.getNextInput(), data.getNextTarget() });
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

} // namespace ANN