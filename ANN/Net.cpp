#include "Net.h"
#include <iostream>
#include <cassert>
#include <fstream>

using namespace ANN;

double Ann::averageSmoothingFactor_ = 100.0; // Number of training samples to average over

Ann::Ann(const std::vector<int>& topology)
    : error_(0.0),
    averageError_(0.2),
    topology_(topology)
{
    int neuronsCount = 0;
    for (size_t i = 0; i < topology_.size(); ++i)
        neuronsCount += topology_[i] + 1;

    bool isBias;
    int  index = 0;
    int  n = 0;
    std::vector<int> in;
    std::vector<int> out;

    for (int L = 0; L < topology_.size(); ++L)
    {
        isBias = false;
        index += topology_[L] + 1;
        for (; n < neuronsCount - 1; ++n)
        {
            if (L == 0 || L != topology_.size() - 1)
            {
                for (int i = 0; i < topology_[L + 1]; ++i)
                    out.emplace_back(index + i);
            }
            if ((L == topology_.size() - 1 || L != 0) && (n != index - 1))
            {
                for (int i = 0; i < topology_[L - 1] + 1; ++i)
                    in.emplace_back(index - topology_[L] - topology_[L - 1] - 2 + i);
            }

            if (n == index - 1)
                isBias = true;
            net_.emplace_back(Neuron(in, out, isBias));
            in.clear();
            out.clear();

            if (n == index - 1)
            {
                ++n;
                break;
            }
        }
    }
}

void Ann::feedForw(const std::vector<double>& worldInput)
{
    assert(worldInput.size() == topology_[INPUT]);

    // Assign world input values to input neurons
    for (int n = 0; n < worldInput.size(); ++n)
        net_[n].setOutput(worldInput[n]);

    // Forward propagate: activate every neuron
    // in every layer except input layer neurons
    for (int n = 0; n < net_.size(); ++n)
        net_[n].activate(net_, n);
}

void Ann::backProp(const std::vector<double>& target)
{
    // Calculate overall net error (RMS of output neuron errors)
    int neuronsCount = 0;
    for (size_t i = 0; i < topology_.size(); ++i)
        neuronsCount += topology_[i] + 1;

    error_ = 0.0; // reset for each backpropagation
    int n = neuronsCount - topology_[topology_.size() - 1] - 1;
    int m = 0;
    for (; n < net_.size(); ++n)
    {
        double delta = target[m++] - net_[n].output();
        error_ += delta * delta;
    }

    error_ /= topology_[topology_.size() - 1]; // average
    error_ = sqrt(error_);                     // RMS

    // Implement a recent average measurement [for displaying, not related to learning]
    averageError_ = (averageError_ * averageSmoothingFactor_ + error_)
                                  / (averageSmoothingFactor_ + 1.0);

    // Calculate hidden layer gradients
    n = neuronsCount - topology_[topology_.size() - 1] - 1;
    m = 0;
    for (; n < net_.size(); ++n)
        net_[n].calcOutputGradients(target[m++]);

    // Calculate hidden layer gradients   
    for (int n = net_.size() - topology_[topology_.size() - 1] - 1; n > topology_[0]; --n)
        net_[n].calcHiddenGradients(net_);

    // Update connection weights
    // for all layers from output to first hidden layer
    for (int n = 0; n < net_.size() - topology_[topology_.size() - 1]; ++n)
        net_[n].updateInputWeights(net_);
}

std::vector<double> Ann::getOutput() const
{
    std::vector<double> result;

    int neuronsCount = 0;
    for (size_t i = 0; i < topology_.size(); ++i)
        neuronsCount += topology_[i] + 1;

    int n = neuronsCount - topology_[topology_.size() - 1] - 1;
    for (; n < net_.size(); ++n)
        result.emplace_back(net_[n].output());

    return result;
}

void Ann::reportState(const std::string& fileName)
{
    std::stringstream ss;
    for (int n = 0; n < net_.size(); ++n)
        net_[n].reportState();

    ss << "===================================="
       << std::endl;

    std::ofstream(fileName) << ss.rdbuf();
}
