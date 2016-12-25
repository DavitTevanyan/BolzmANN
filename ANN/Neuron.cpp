#include "Neuron.h"
#include <iostream>

using namespace ANN;

double Neuron::rate     = 0.5;  // overall net learning rate,                range [0.0, 1.0]
double Neuron::momentum = 0.5;  // momentum, multiplier of last deltaWeight, range [0.0, 1.0]

Neuron::Neuron(const std::vector<int>& ins, const std::vector<int>& outs, bool isBias)
    : output_(0.0),
    gradient_(0.0),
    ins_(ins),
    outs_(outs),
    isBias_(isBias)
{
    if (isBias_)
        output_ = 1.0;

    for (int i = 0; i < outs_.size(); ++i)
        axon_.emplace_back(Connection());
}

void Neuron::activate(const std::vector<Neuron>& net, int n)
{
    if (ins_.empty()) return;

    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs),
    // including the bias node of the previous layer
    std::ptrdiff_t index = 0;
    for (const auto& i : ins_)
    {
        index = std::find(net[i].outs_.begin(), net[i].outs_.end(), n) - net[i].outs_.begin();
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
    for (int n = 0; n < outs_.size(); ++n)
        sum += axon_[n].weight * net[outs_[n]].gradient_;

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
    for (int n = 0; n < outs_.size(); ++n)
    {
        double oldDeltaWeight = axon_[n].deltaWeight;
        double newDeltaWeight = rate * net[outs_[n]].gradient_ * output_
                              + momentum * oldDeltaWeight;

        axon_[n].deltaWeight = newDeltaWeight;
        axon_[n].weight     += newDeltaWeight;
    }
}

void Neuron::addIn()
{
    ins_.emplace_back(ins_.back() + 1);
}

void Neuron::addOut(int pos)
{
    outs_.emplace_back(outs_.back() + 1);
    axon_.insert(axon_.begin() + pos, Connection());
}

void Neuron::updateInsOuts(int index, bool operation)
{
    for (auto& i : ins_)
        if (i > index)
            operation ? ++i : --i;

    for (auto& o : outs_)
        if (o > index)
            operation ? ++o : --o;
}

void Neuron::removeOuts(std::vector<Neuron>& net, int index)
{
    for (auto& n : net)
    {
        auto it = std::find(n.outs_.begin(), n.outs_.end(), index);
        if (it != n.outs_.end())
        {
            std::ptrdiff_t pos = it - n.outs_.begin();
            auto itWeight = n.axon_.begin();
            for (int i = 0; i < pos; ++i)
                ++itWeight;
            int m = 0;
            for (; it != n.outs_.end();)
            {
                if (*it == index)
                {
                    it = n.outs_.erase(it);  
                    ++m;
                }
                else
                   ++it;
            }
            for (int i = 0; i < m; ++i)
                itWeight = n.axon_.erase(itWeight);
        }
    }
}

void Neuron::removeIns(std::vector<Neuron>& net, int index)
{
    for (auto& n : net)
    {
        auto it = std::find(n.ins_.begin(), n.ins_.end(), index);
        for (; it != n.ins_.end();)
            (*it == index) ? it = n.ins_.erase(it) : ++it;
    }
}

void Neuron::addConnection(int index, bool direction)
{
    if (direction)
        ins_.insert(std::upper_bound(ins_.cbegin(), ins_.cend(), index), index);
    else
    {
        auto itIndex = std::upper_bound(outs_.cbegin(), outs_.cend(), index);
        std::ptrdiff_t pos = itIndex - outs_.begin();
        outs_.insert(itIndex, index);

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
        auto it = std::find(ins_.begin(), ins_.end(), index);
        if (it == ins_.end())
            throw std::out_of_range("Cannot remove connection, because it doesn't exist.");
            
        ins_.erase(it);
    }
    else
    {
        auto it = std::find(outs_.begin(), outs_.end(), index);
        if (it == outs_.end())
            throw std::out_of_range("Cannot remove connection, because it doesn't exist.");

        std::ptrdiff_t pos = it - outs_.begin();
        outs_.erase(it);

        auto itWeight = axon_.begin();
        for (int i = 0; i < pos; ++i)
            ++itWeight;
        axon_.erase(itWeight);
    }
}

std::string Neuron::reportState()
{
    std::string neuron = "--------------\n";

    for (size_t n = 0; n < outs_.size(); ++n)
    {
        neuron += "W_" + std::to_string(n + 1) + "  " + std::to_string(axon_[n].weight) + "\n"
                + "D_" + std::to_string(n + 1) + "  " + std::to_string(axon_[n].weight) + "\n";
    }
    neuron += "G    " + std::to_string(gradient_) + "\n"
            + "O    " + std::to_string(output_)   + "\n";

    neuron += "--------------\n";
    return neuron;
}
