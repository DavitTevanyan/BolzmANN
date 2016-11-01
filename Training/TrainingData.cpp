#include "TrainingData.h"
#include <sstream>

using namespace ANN;

std::vector<int> TrainingData::getTopology()
{
    std::vector<int> topology;
    std::string      line;

    getline(trainingDataFile_, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;
    if (isEof() || label != "topology:") 
    {
        abort();
    }

    while (!ss.eof()) 
    {
        int n;
        ss >> n;
        topology.emplace_back(n);
    }

    return topology;
}

TrainingData::TrainingData(const std::string& filename)
{
    trainingDataFile_.open(filename.c_str());
}

int TrainingData::getNextInput(std::vector<double>& inputVals)
{
    inputVals.clear();

    std::string line;
    getline(trainingDataFile_, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;
    if (label == "in:")
    {
        double oneValue;
        while (ss >> oneValue) 
        {
            inputVals.emplace_back(oneValue);
        }
    }

    return static_cast<int>(inputVals.size());
}

std::vector<double> TrainingData::getTargetOutput()
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