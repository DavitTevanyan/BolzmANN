#include "Neuron.h"

using namespace ANN;

double Neuron::eta   = 0.15;  // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5;   // momentum, multiplier of last deltaWeight, [0.0..1.0]

void Neuron::updateInputWeights(Layer& prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for (auto& neuron : prevLayer)
    {
        double oldDeltaWeight = neuron.connections_[id_].deltaWeight;
        double newDeltaWeight = eta * neuron.outputVal() * gradient_ // Individual input, magnified by the gradient and train rate:
                              + alpha * oldDeltaWeight;                 // Also add momentum = a fraction of the previous delta weight;

        neuron.connections_[id_].deltaWeight = newDeltaWeight;
        neuron.connections_[id_].weight     += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.
    for (int n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += connections_[n].weight * nextLayer[n].gradient_;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
    const double dow = sumDOW(nextLayer);
    gradient_  = dow * Neuron::transferFunctionDerivative(outputVal_);
}

void Neuron::calcOutputGradients(const double targetVal)
{
    const double delta = targetVal - outputVal_;
    gradient_  = delta * Neuron::transferFunctionDerivative(outputVal_);
}

double Neuron::transferFunction(const double x)
{
    return tanh(x);     // tanh - output range [-1.0..1.0]
}

double Neuron::transferFunctionDerivative(const double x)
{
    return 1.0 - x * x; // tanh derivative
}

void Neuron::feedForward(const Layer& prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.
    for (const auto& neuron : prevLayer) 
    {
        sum += neuron.outputVal() * neuron.connections_[id_].weight;
    }

    outputVal_ = Neuron::transferFunction(sum);
}

Neuron::Neuron(int numOutputs, int myId) : outputVal_(0.0), gradient_(0.0)
{
    for (int i = 0; i < numOutputs; ++i)
    {
        connections_.emplace_back(Connection());
        connections_.back().weight = randomWeight();
    }

    id_ = myId;
}
