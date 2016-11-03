#include "TrainingData.h"
#include "SpecificUtilities.h"
#include "Net.h"

using namespace ANN;

int main()
{
    Net net({ 3, 2, 1 }); // topology described by the initializer-list

    for (int i = 0; i < 200; i++)
    {
        TrainingData trainData("trainingData.txt");

        while (!trainData.isEof())
        {
            const std::vector<double> input  = trainData.getNextInput();
            net.feedForw(input);
            const std::vector<double> output = net.getResult();
            const std::vector<double> target = trainData.getTargetOutput();

            if (!target.empty())
            {
                static double pass = 1; // double to prevent narrowing conversion below
                display("Pass:", { pass++ }, true);

                net.backProp(target); // train on what the outputs should have been

                display("Input: ", input);
                display("Target:", target);
                display("Output:", output);

                // Report how well the training is working, average over recent samples
                std::cout << "-----------------------------------"                       << std::endl;
                std::cout << "Net recent average error: " << net.getRecentAverageError() << std::endl;
                std::cout << "-----------------------------------"                       << std::endl;
            }            
        }
    }

    std::cout << std::endl << "Done" << std::endl;
}
