#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include <vector>
#include <fstream>
#include <string>

namespace ANN {

    static double pass = 1;

    struct Sample {
        std::vector<double> input;
        std::vector<double> target;
    };

    std::vector<Sample> getTrainSet();

    class TrainingData
    {
    public:
        TrainingData(const std::string& filename);

        std::vector<double> getNextInput();
        std::vector<double> getNextTarget();

        bool allRead() { return trainingDataFile_.eof(); }

    private:
        std::ifstream trainingDataFile_;
    };

    extern std::vector<Sample> AND;

} // namespace ANN

#endif // TRAININGDATA_H

