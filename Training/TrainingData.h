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

        std::vector<double> getNextInput();
        std::vector<double> getTargetOutput();

        bool isEof() { return trainingDataFile_.eof(); }

    private:
        std::ifstream trainingDataFile_;
    };
}

#endif // TRAININGDATA_H

