#include "Net.h"
#include <iostream>
#include <cassert>
#include <fstream>

using namespace ANN;

double Ann::averageSmoothingFactor_ = 100.0; // Number of training samples to average over

Ann::Ann(const std::vector<int>& layers)
    : error_(0.0), averageError_(0.2)
{
    const size_t numLayers = layers.size();

    for (int L = 0; L < numLayers; ++L)
    {
        layers_.emplace_back(Layer());

        int outs = 0;
        if (L < numLayers - 1)
            outs = layers[L + 1];

        // Fill layer with neurons; last neuron is bias
        for (int posL = 0; posL <= layers[L]; ++posL)
        {
            layers_[L].emplace_back(Neuron(posL, outs));
        }

        // Bias is last neuron, fixed output
        layers_[L].back().setOutput(1.0); // completely irrelevant for L == numLayers-1
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
    for (int L = 1; L < layers_.size(); ++L)
    {
        Layer& prevLayer  = layers_[L - 1];
        for (int n = 0; n < layers_[L].size() - 1; ++n)
        {
            layers_[L][n].activate(prevLayer);
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
        Layer& hiddLayer = layers_[i];
        Layer& nextLayer = layers_[i + 1];

        for (auto& neuron : hiddLayer) 
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

void Ann::reportState(const std::string& fileName)
{
    std::stringstream ss;
    for (size_t i = 0; i < layers_.size() - 1; ++i)
    {
        Layer& currentLayer = layers_[i];
        Layer& nextLayer    = layers_[i + 1];
        for (auto& neuron : currentLayer)
        {
            ss << neuron.reportState(nextLayer);
        }
        ss << "===================================="
           << std::endl;
    }

    std::ofstream(fileName) << ss.rdbuf();
}
