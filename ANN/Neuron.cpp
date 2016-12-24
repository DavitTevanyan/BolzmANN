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
    std::ptrdiff_t index = 0;
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

void Neuron::addIns()
{
    in_.emplace_back(in_.back() + 1);
}

void Neuron::addOuts(int pos)
{
    out_.emplace_back(out_.back() + 1);
    axon_.insert(axon_.begin() + pos, Connection());
}

void Neuron::removeOuts(std::vector<Neuron>& net, int index)
{
    for (auto& n : net)
    {
        auto it = std::find(n.out_.begin(), n.out_.end(), index);
        if (it != n.out_.end())
        {
            std::ptrdiff_t pos = it - n.out_.begin();
            auto itWeight = n.axon_.begin();
            for (int i = 0; i < pos; ++i)
                ++itWeight;
            int m = 0;
            for (; it != n.out_.end();)
            {
                if (*it == index)
                {
                    it = n.out_.erase(it);  
                    ++m;
                }
                else
                   ++it;
            }
            for (int i = 0; i < m; ++i)
                itWeight = n.axon_.erase(itWeight);
        }
        for (auto& i : n.out_)
            if (i > index)
                --i;
    }
}

void Neuron::removeIns(std::vector<Neuron>& net, int index)
{
    for (auto& n : net)
    {
        auto it = std::find(n.in_.begin(), n.in_.end(), index);
        for (; it != n.in_.end();)
            (*it == index) ? it = n.in_.erase(it) : ++it;

        for (auto& i : n.in_)
            if (i > index)
                --i;
    }
}

void Neuron::updateIns(bool operation)
{
    if (operation)
    {
        for (int i = 0; i < in_.size(); ++i)
            ++in_[i];
    }
    else
    {

    }
}

void Neuron::updateOuts(bool operation)
{
    if (operation)
    {
        for (int i = 0; i < out_.size(); ++i)
            ++out_[i];
    }
    else
    {

    }
}

void Neuron::addConnection(int index, bool direction)
{
    if (direction)
        in_.insert(std::upper_bound(in_.cbegin(), in_.cend(), index), index);
    else
    {
        auto itIndex = std::upper_bound(out_.cbegin(), out_.cend(), index);
        std::ptrdiff_t pos = itIndex - out_.begin();
        out_.insert(itIndex, index);

        auto itWeight = axon_.begin();
        for (int i = 0; i < pos; ++i)
            ++itWeight;
        axon_.insert(itWeight, Connection());
    }
}

void Neuron::deleteConnection(int index, bool direction)
{
    if (direction)
    {
        auto it = std::find(in_.begin(), in_.end(), index);
        if (it == in_.end())
            throw std::out_of_range("Cannot remove connection, because it doesn't exist.");
            
        in_.erase(it);
    }
    else
    {
        auto it = std::find(out_.begin(), out_.end(), index);
        if (it == out_.end())
            throw std::out_of_range("Cannot remove connection, because it doesn't exist.");

        std::ptrdiff_t pos = it - out_.begin();
        out_.erase(it);

        auto itWeight = axon_.begin();
        for (int i = 0; i < pos; ++i)
            ++itWeight;
        axon_.erase(itWeight);
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
