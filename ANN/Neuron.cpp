#include "Neuron.h"

using namespace ANN;

double Neuron::eta   = 0.15;  // overall net learning rate,                range [0.0, 1.0]
double Neuron::alpha = 0.5;   // momentum, multiplier of last deltaWeight, range [0.0, 1.0]

Neuron::Neuron(int numOutputs, int myId)
    : output_(0.0), gradient_(0.0), id_(myId)
{
    for (int i = 0; i < numOutputs; ++i)
    {
        connections_.emplace_back(Connection());
    }
}

void Neuron::activate(const Layer& prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.
    for (const auto& neuron : prevLayer)
    {
        sum += neuron.getOutput() * neuron.connections_[id_].weight; // TODO: connections_[id_] inelegant
    }

    output_ = activationFunction(sum);
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
    // Since we don't have a target value to compare with
    // for a hidden neuron, we take something equivalent: DOW:
    // sum of the derivatives of the weights of the next layer 
    const double dow = sumDOW(nextLayer);
    gradient_  = dow * activationFunctionDerivative(output_);
}

void Neuron::calcOutputGradients(const double targetVal)
{
    const double delta = targetVal - output_;
    gradient_  = delta * activationFunctionDerivative(output_);
}

double Neuron::activationFunction(const double x)
{
    return tanh(x); // gives an output range of [-1.0, 1.0]
}

double Neuron::activationFunctionDerivative(const double x)
{
    return 1.0 - x * x; // tanh derivative (quick approximation)
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for (auto& neuron : prevLayer)
    {
        double oldDeltaWeight = neuron.connections_[id_].deltaWeight;
        double newDeltaWeight = eta * neuron.getOutput() * gradient_ // individual input, magnified by the gradient and train rate;
                              + alpha * oldDeltaWeight;              // also add momentum = a fraction of the previous delta weight;

        neuron.connections_[id_].deltaWeight = newDeltaWeight;
        neuron.connections_[id_].weight     += newDeltaWeight;
    }
}

// Sum of the derivatives of the weights of the next layer
double Neuron::sumDOW(const Layer& nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions to the errors of the nodes we feed
    for (int n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += connections_[n].weight * nextLayer[n].gradient_;
    }

    return sum;
}