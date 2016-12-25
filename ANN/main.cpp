#include "SpecificUtilities.h"
#include "TrainingData.h"
#include "Net.h"

using namespace ANN;

int main() try
{
    std::vector<Sample> trainSet = getTrainSet("../Training/and.txt");

    Ann ann({ 2, 2, 3, 1 }); // topology by initializer-list

    ann.addNeuron(2, 2, false);
    //ann.deleteNeuron(2, 1);
    //ann.deleteNeuron(2, 1);
    //ann.deleteNeuron(2, 1);
    
    //ann.deleteConnection(2, 2, 4, 1);
    //ann.addConnection(1, 2, 3, 1);

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
catch (const std::exception& e)
{
    display(e.what());
}
catch (...)
{
    display("ERROR: Unknown.");
}
