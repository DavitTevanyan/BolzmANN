#include "TrainingData.h"
#include "Net.h"
#include <iostream>
#include <cassert>

using namespace ANN;

void display(const std::string& label, const std::vector<double>& v)
{
    std::cout << label << " ";

    for (const auto& elem : v) 
    {
        std::cout << elem << " ";
    }

    std::cout << std::endl;
}

int main()
{
    TrainingData trainData("trainingData.txt");

    // e.g., { 3, 2, 1 }
    std::vector<int> topology;
    topology = trainData.getTopology();

    Net net(topology);

    std::vector<double> inputVals; 
    std::vector<double> targetVals;

    int pass = 0;
    while (!trainData.isEof()) 
    {
        std::cout << std::endl << "Pass:    " << ++pass << std::endl;

        // Get new input data and feed it forward
        if (trainData.getNextInput(inputVals) != topology[0]) 
        {
            break;
        }
        net.feedForward(inputVals);

        display("Inputs: ", inputVals);        
        display("Outputs:", net.getResult());

        // Train the net what the outputs should have been
        trainData.getTargetOutput(targetVals);
        net.backProp(targetVals); // actual train
        display("Targets:", targetVals);
        
        // Report how well the training is working, average over recent samples
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "Net recent average error: " << net.getRecentAverageError() << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }

    std::cout << std::endl << "Done" << std::endl;
}
