#include "Net.h"

using namespace ANN;

int main() try
{
    std::vector<Sample> trainSet = getTrainSet("../Training/and.txt");

    Ann ann({ 2, 2, 3, 1 }); // topology by initializer-list

    //ann.addNeuron({2, 2}, false);
    //ann.deleteNeuron({2, 1});
    //ann.deleteNeuron({2, 1});
    //ann.deleteNeuron({2, 1});
    
    //ann.deleteConnection({2, 2}, {4, 1});
    //ann.addConnection({1, 2}, {3, 1});

    // Train
    const double avrgError = 0.005;
    ann.trainNet(trainSet, avrgError);

    // Test
    ann.testNet(trainSet);

    displayStats(ann.averageError(), pass);
    ann.reportState("NetState");
}
catch (const std::exception& e)
{
    display(e.what());
}
catch (...)
{
    display("ERROR: Unknown.");
}
