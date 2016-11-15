#include "SpecificUtilities.h"
#include "TrainingData.h"
#include "Net.h"

using namespace ANN;

int main()
{
    TrainingData data("trainingData.txt");
    std::vector<Data> trainSets;
    while (!data.isEof())
        trainSets.push_back({ data.getNextInput(), data.getNextTarget() });

    Ann ann({ 2, 2, 1 }); // topology described by initializer-list
    
    while (ann.averageError() > 0.05)
    {
        for (const auto& set : trainSets)
        {
            ann.feedForw(set.input);
            ann.backProp(set.target);

            display("Pass:", { pass++ }, alignRight);
            display("Input: ", set.input);
            display("Target:", set.target);
            display("Output:", ann.getOutput());
            displayNetError(ann.averageError());
        }
    }

//    display("Pass:", { pass }, alignRight);
//    displayNetError(ann.averageError());

    std::cout << std::endl << "Done" << std::endl;
}
