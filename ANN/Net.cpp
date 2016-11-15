#include "Net.h"
#include <iostream>
#include <cassert>

using namespace ANN;

double Ann::averageSmoothingFactor_ = 100.0; // Number of training samples to average over

Ann::Ann(const std::vector<int>& topology)
    : error_(0.0), averageError_(5.0)
{
    const size_t numLayers = topology.size();

    for (int i = 0; i < numLayers; ++i)
    {
        layers_.emplace_back(Layer());
        const int neuronOutputs = (i == topology.size() - 1) ? 0 : topology[i + 1]; // fully connected net

        // Fill layer with neurons; last neuron is bias
        const auto L = topology[i];
        for (int idxL = 0; idxL <= L; ++idxL)
        {
            layers_.back().emplace_back(Neuron(neuronOutputs, idxL));
        }

        // Bias is last neuron, fixed output
        layers_.back().back().setOutput(1.0);
    }
}

void Ann::feedForw(const std::vector<double>& worldInput)
{
    assert(worldInput.size() == layers_[INPUT].size() - 1);

    // Assign world input values to input neurons
    for (int n = 0; n < worldInput.size(); ++n)
    {
        layers_[INPUT][n].setOutput(worldInput[n]);
    }

    // Forward propagate: activate every neuron
    // in every layer except input layer neurons
    for (int i = 1; i < layers_.size(); ++i)
    {
        Layer& prevLayer  = layers_[i - 1];
        for (int n = 0; n < layers_[i].size() - 1; ++n)
        {
            layers_[i][n].activate(prevLayer);
        }
    }
}

void Ann::backProp(const std::vector<double>& target)
{
    // Calculate overall net error (RMS of output neuron errors)
    auto& outputLayer = layers_.back();
    error_ = 0.0; // reset for each backpropagation

    for (int n = 0; n < outputLayer.size() - 1; ++n) 
    {
        double delta = target[n] - outputLayer[n].output();
        error_ += delta * delta;
    }
    error_ /= outputLayer.size() - 1; // average
    error_ = sqrt(error_);            // RMS

    // Implement a recent average measurement [for displaying, not related to learning]
    averageError_ = (averageError_ * averageSmoothingFactor_ + error_)
                  / (                averageSmoothingFactor_ + 1.0);

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
        Layer& prevLayer = layers_[i - 1];

        for (int n = 0; n < layers_[i].size() - 1; ++n) 
        {
            layers_[i][n].updateInputWeights(prevLayer);
        }
    }
}

std::vector<double> Ann::getOutput() const
{
    std::vector<double> result;

    for (const auto& neuron : layers_.back())
        result.push_back(neuron.output());

    result.pop_back(); // remove bias neuron

    return result;
}
