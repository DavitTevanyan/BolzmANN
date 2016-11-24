#include "Neuron.h"
#include <iostream>

using namespace ANN;

double Neuron::rate     = 0.6;  // overall net learning rate,                range [0.0, 1.0]
double Neuron::momentum = 0.4;  // momentum, multiplier of last deltaWeight, range [0.0, 1.0]

    : output_(0.0), gradient_(0.0), posL_(posL)
{
    for (int i = 0; i < outs; ++i)
    {
        axon_.emplace_back(Connection());
    }
}

void Neuron::activate(const Layer& prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs),
    // including the bias node of the previous layer
    for (const auto& neuron : prevLayer)
        sum += neuron.output() * neuron.axon_[posL_].weight;

    output_ = af(sum);;
}

void Neuron::calcOutputGradients(const double target)
{
    const double delta = target - output_;
    gradient_  = delta * af_Derivative(output_);
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
    // Since we don't have a target value to compare with
    // for a hidden neuron, we take something equivalent: DOW:
    // sum of the derivatives of the weights of the next layer 
    const double dow = sumDOW(nextLayer);
    gradient_  = dow * af_Derivative(output_);
}

double Neuron::af(double x)
{
    return tanh(x); // gives an output range of [-1.0, 1.0]
}

double Neuron::af_Derivative(const double x)
{
    return 1.0 - x * x; // tanh derivative (quick approximation)
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for (auto& neuron : prevLayer)
    {
        double oldDeltaWeight = neuron.axon_[posL_].deltaWeight;
        double newDeltaWeight = rate * gradient_ * neuron.output()
                              + momentum * oldDeltaWeight; // add a fraction of the old delta weight

        neuron.axon_[posL_].deltaWeight = newDeltaWeight;
        neuron.axon_[posL_].weight     += newDeltaWeight;
    }
}

// Sum of the derivatives of the weights of the next layer
double Neuron::sumDOW(const Layer& nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions to the errors of the nodes we feed
    for (int n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += axon_[n].weight * nextLayer[n].gradient_;
    }

    return sum;
}

std::string Neuron::reportState(Layer& nextLayer)
{
    std::string neuron = "--------------\n";

    for (size_t n = 0; n < nextLayer.size() - 1; ++n)
    {
        neuron += "W_" + std::to_string(n + 1) + "  " + std::to_string(axon_[n].weight) + "\n"
                + "D_" + std::to_string(n + 1) + "  " + std::to_string(axon_[n].weight) + "\n";
    }
    neuron +=  "G    " + std::to_string(gradient_) + "\n"
            +  "O    " + std::to_string(output_)   + "\n";

    return neuron += "--------------\n";
}
