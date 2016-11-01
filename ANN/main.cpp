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

    std::vector<double> input; 
    std::vector<double> target;

    int pass = 0;
    while (!trainData.isEof()) 
    {
        std::cout << std::endl << "Pass:    " << ++pass << std::endl;

        // Get new input data and feed it forward
        if (trainData.getNextInput(input) != topology[0]) 
        {
            break;
        }
        net.feedForw(input);

        display("Inputs: ", input);        
        display("Outputs:", net.getResult());

        // Train the net what the outputs should have been
        auto target = trainData.getTargetOutput();
        net.backProp(target);         // actual training
        display("Targets:", target);
        
        // Report how well the training is working, average over recent samples
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "Net recent average error: " << net.getRecentAverageError() << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }

    std::cout << std::endl << "Done" << std::endl;
}
