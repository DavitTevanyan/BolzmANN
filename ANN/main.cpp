#include "SpecificUtilities.h"
#include "TrainingData.h"
#include "Net.h"

using namespace ANN;

int main()
{
    std::vector<Sample> trainSet = getTrainSet("../Training/and.txt");

    Ann ann({ 2, 2, 3, 1 }); // topology by initializer-list
    
    // Train
    while (ann.averageError() > 0.05)
    {
        for (const auto& sample : trainSet)
        {
            pass++;
            ann.feedForw(sample.input);
            ann.backProp(sample.target);
        }
    }

    // Test
    for (const auto& sample : trainSet)
    {
        ann.feedForw(sample.input);

        display("Input: ", sample.input);
        display("Target:", sample.target);
        display("Output:", ann.getOutput());
    }

    displayStats(ann.averageError(), pass);
}
