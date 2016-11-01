#include "Net.h"
#include <iostream>
#include <cassert>

using namespace ANN;

double Net::recentAverageSmoothingFactor_ = 100.0; // Number of training samples to average over

Net::Net(const std::vector<int>& topology)
    : error_(0.0), recentAverageError_(0.0)
{
    size_t numLayers = topology.size();

    for (int i = 0; i < numLayers; ++i)
    {
        layers_.emplace_back(Layer());
        const int numOutputs = (i == topology.size() - 1) ? 0 : topology[i + 1];

        // Fill layer with neurons; last neuron is bias
        const auto L = topology[i];
        for (int id = 0; id <= L; ++id)
        {
            layers_.back().emplace_back(Neuron(numOutputs, id));
        }

        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer)
        layers_.back().back().setOutput(1.0);
    }
}

void Net::feedForw(const std::vector<double>& worldInput)
{
    assert(worldInput.size() == layers_[0].size() - 1);

    // Assign world input values to input neurons
    for (int i = 0; i < worldInput.size(); ++i)
    {
        layers_[0][i].setOutput(worldInput[i]);
    }

    // Forward propagate
    for (int i = 1; i < layers_.size(); ++i)
    {
        Layer& prevLayer = layers_[i - 1];
        for (int n = 0; n < layers_[i].size() - 1; ++n)
        {
            layers_[i][n].activate(prevLayer);
        }
    }
}

void Net::backProp(const std::vector<double>& target)
{
    // Calculate overall net error (RMS of output neuron errors)
    auto& outputLayer = layers_.back();
    error_ = 0.0;

    for (int n = 0; n < outputLayer.size() - 1; ++n) 
    {
        double delta = target[n] - outputLayer[n].getOutput();
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
        outputLayer[n].calcOutputGradients(target[n]);
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

    // Update connection weights
    // for all layers from output to first hidden layer
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

std::vector<double> Net::getResult() const
{
    std::vector<double> result;

    const auto& backLayer = layers_.back();

    auto   itNeuron  = backLayer.begin();
    for (; itNeuron != backLayer.end() - 1; ++itNeuron)
    {
        result.emplace_back(itNeuron->getOutput());
    }
    return result;
}
