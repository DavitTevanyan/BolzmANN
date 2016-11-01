#include "Net.h"
#include <iostream>
#include <cassert>

using namespace ANN;

double Net::recentAverageSmoothingFactor_ = 100.0; // Number of training samples to average over

std::vector<double> Net::getResult() const
{
    std::vector<double> result;

    const auto&  backLayer = layers_.back();

    auto   it  = backLayer.begin();
    for (; it != backLayer.end() - 1; ++it)
    {
        result.emplace_back(it->outputVal());
    }
    return result;
}
void Net::backProp(const std::vector<double>& targetVals)
{
    // Calculate overall net error (RMS of output neuron errors)
    auto& outputLayer = layers_.back();
    error_ = 0.0;

    for (int n = 0; n < outputLayer.size() - 1; ++n) 
    {
        double delta = targetVals[n] - outputLayer[n].outputVal();
        error_ += delta * delta;
    }
    error_ /= outputLayer.size() - 1; // get average error squared
    error_ = sqrt(error_);            // RMS

    // Implement a recent average measurement
    recentAverageError_ = (recentAverageError_ * recentAverageSmoothingFactor_ + error_)
                        / (                      recentAverageSmoothingFactor_ + 1.0);

    // Calculate output layer gradients
    for (int n = 0; n < outputLayer.size() - 1; ++n) 
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate hidden layer gradients
    for (size_t i = layers_.size() - 2; i > 0; --i) 
    {
        Layer& hiddenLayer = layers_[i];
        Layer& nextLayer   = layers_[i + 1];

        for (auto& neuron : hiddenLayer) 
        {
            neuron.calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights
    for (size_t i = layers_.size() - 1; i > 0; --i) 
    {
        Layer& layer     = layers_[i];
        Layer& prevLayer = layers_[i - 1];

        for (int n = 0; n < layer.size() - 1; ++n) 
        {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const std::vector<double>& inputVals)
{
    assert(inputVals.size() == layers_[0].size() - 1);

    // Assign (latch) the input values into the input neurons
    for (int i = 0; i < inputVals.size(); ++i) 
    {
        layers_[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagate
    for (int i = 1; i < layers_.size(); ++i) 
    {
        Layer& prevLayer  = layers_[i - 1];
        for (int n = 0; n < layers_[i].size() - 1; ++n) 
        {
            layers_[i][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const std::vector<int>& topology) : error_(0.0), recentAverageError_(0.0)
{
    size_t numLayers = topology.size();

    for (int i = 0; i < numLayers; ++i) 
    {
        layers_.emplace_back(Layer());
        const int numOutputs = i == topology.size() - 1 ? 0 : topology[i + 1];

        // Fill layer with neurons and add a bias neuron in each layer
        const auto L = topology[i];
        for (int neuronId = 0; neuronId <= L; ++neuronId)
        {
            layers_.back().emplace_back(Neuron(numOutputs, neuronId));
            std::cout << "Neuron made." << std::endl;
        }

        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer)
        layers_.back().back().setOutputVal(1.0);
    }
}
