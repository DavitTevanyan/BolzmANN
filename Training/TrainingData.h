#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include <vector>
#include <fstream>
#include <string>

namespace ANN
{
    static double pass = 1;

    struct Data {
        std::vector<double> input;
        std::vector<double> target;
    };

    class TrainingData
    {
    public:
        explicit TrainingData(const std::string& filename);

        std::vector<double> getNextInput();
        std::vector<double> getNextTarget();

        bool isEof() { return trainingDataFile_.eof(); }

    private:
        std::ifstream trainingDataFile_;
    };
}

#endif // TRAININGDATA_H

