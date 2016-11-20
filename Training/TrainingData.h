#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include <fstream>
#include <vector>
#include <string>

namespace ANN {

    static double pass = 1;

    struct Sample {
        std::vector<double> input;
        std::vector<double> target;
    };

    std::vector<Sample> getTrainSet(const std::string& fileName);

    class TrainingData
    {
    public:
        TrainingData(const std::string& filename);
        std::vector<double> getValues();
        bool                allRead() { return dataFile_.eof(); }

    private:
        std::ifstream dataFile_;
    };

///////////////////////////////////////////////////////
// Samples for profiling (to avoid slow file access) //
///////////////////////////////////////////////////////

    extern std::vector<Sample> AND;
    extern std::vector<Sample> XOR;

} // namespace ANN

#endif // TRAININGDATA_H

