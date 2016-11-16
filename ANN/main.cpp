#include "SpecificUtilities.h"
#include "TrainingData.h"
#include "Net.h"

using namespace ANN;

int main()
{
    auto trainSet = getTrainSet();

    Ann ann({ 2, 2, 3, 1 }); // topology by initializer-list
    
    while (ann.averageError() > 0.05)
    {
        for (const auto& sample : trainSet)
        {
            pass++;
            ann.feedForw(sample.input);
            ann.backProp(sample.target);
        }
    }

    for (const auto& sample : trainSet)
    {
        ann.feedForw(sample.input);

        display("Input: ", sample.input);
        display("Target:", sample.target);
        display("Output:", ann.getOutput());
    }

    displayStats(ann.averageError(), pass);
}
