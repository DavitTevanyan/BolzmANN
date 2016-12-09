#include "Neuron.h"
#include <iostream>

using namespace ANN;

double Neuron::rate     = 0.5;  // overall net learning rate,                range [0.0, 1.0]
double Neuron::momentum = 0.5;  // momentum, multiplier of last deltaWeight, range [0.0, 1.0]

Neuron::Neuron(const std::vector<int>& in, const std::vector<int>& out, bool isBias)
    : output_(0.0),
    gradient_(0.0),
    in_(in),
    out_(out),
    isBias_(isBias)
{
    if (isBias_)
        output_ = 1.0;

    for (int i = 0; i < out_.size(); ++i)
        axon_.emplace_back(Connection());
}

void Neuron::activate(const std::vector<Neuron>& net, int n)
{
    if (in_.empty()) return;

    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs),
    // including the bias node of the previous layer
    int index = 0;
    for (const auto& i : in_)
    {
        index = std::find(net[i].out_.begin(), net[i].out_.end(), n) - net[i].out_.begin();
        sum += net[i].output() * net[i].axon_[index].weight;
    }

    output_ = af(sum);
}

void Neuron::calcOutputGradients(const double target)
{
    const double delta = target - output_;
    gradient_  = delta * af_Derivative(output_);
}

void Neuron::calcHiddenGradients(const std::vector<Neuron>& net)
{
    if (isBias_) return;

    // Since we don't have a target value to compare with
    // for a hidden neuron, we take something equivalent: DOW:
    // sum of the derivatives of the weights of the next layer 
    double sum = 0.0;

    // Sum our contributions to the errors of the nodes we feed
    for (int n = 0; n < out_.size(); ++n)
        sum += axon_[n].weight * net[out_[n]].gradient_;

    gradient_ = sum * af_Derivative(output_);
}

double Neuron::af(double x)
{
    return tanh(x);     // gives an output range of [-1.0, 1.0]
}

double Neuron::af_Derivative(const double x)
{
    return 1.0 - x * x; // tanh derivative (quick approximation)
}

void Neuron::updateInputWeights(const std::vector<Neuron>& net)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for (int n = 0; n < out_.size(); ++n)
    {
        double oldDeltaWeight = axon_[n].deltaWeight;
        double newDeltaWeight = rate * net[out_[n]].gradient_ * output_
                              + momentum * oldDeltaWeight;

        axon_[n].deltaWeight = newDeltaWeight;
        axon_[n].weight     += newDeltaWeight;
    }
}

std::string Neuron::reportState()
{
    std::string neuron = "--------------\n";

    for (size_t n = 0; n < out_.size(); ++n)
    {
        neuron += "W_" + std::to_string(n + 1) + "  " + std::to_string(axon_[n].weight) + "\n"
                + "D_" + std::to_string(n + 1) + "  " + std::to_string(axon_[n].weight) + "\n";
    }
    neuron += "G    " + std::to_string(gradient_) + "\n"
            + "O    " + std::to_string(output_)   + "\n";

    neuron += "--------------\n";
    return neuron;
}
