#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include <vector>
#include <fstream>
#include <string>

namespace ANN
{
    class TrainingData
    {
    public:
        explicit TrainingData(const std::string& filename);
        std::vector<int> getTopology();
        bool             isEof() { return trainingDataFile_.eof(); }

        // Returns the number of input values read from the file:
        int getNextInput(std::vector<double>& inputVals);
        int getTargetOutput(std::vector<double>& targetOutputVals);

    private:
        std::ifstream trainingDataFile_;
    };
}

#endif // TRAININGDATA_H

