#include "Net.h"
#include <iostream>
#include <cassert>
#include <fstream>

using namespace ANN;

double Ann::averageSmoothingFactor_ = 100.0; // Number of training samples to average over

Ann::Ann(const std::vector<int>& topology)
    : error_(0.0),
    averageError_(0.2),
    topology_(topology)
{
    int neuronsCount = 0;
    for (size_t i = 0; i < topology_.size(); ++i)
        neuronsCount += topology_[i] + 1;

    bool isBias;
    int  index = 0;
    int  n = 0;
    std::vector<int> in;
    std::vector<int> out;

    // Initialize vector net_ with neurons like following structure
    // (neuron, inNeuronsList, outNeuronsList)
    for (int L = 0; L < topology_.size(); ++L)
    {
        isBias = false;
        index += topology_[L] + 1;
        for (; n < neuronsCount - 1; ++n)
        {
            if (L != topology_.size() - 1)
            {
                for (int i = 0; i < topology_[L + 1]; ++i)
                    out.emplace_back(index + i);
            }
            if (L != 0 && n != index - 1)
            {
                for (int i = 0; i < topology_[L - 1] + 1; ++i)
                    in.emplace_back(index - topology_[L] - topology_[L - 1] - 2 + i);
            }

            if (n == index - 1)
                isBias = true;
            net_.emplace_back(Neuron(in, out, isBias));
            in.clear();
            out.clear();

            if (n == index - 1)
            {
                ++n;
                break;
            }
        }
    }
}

void Ann::feedForw(const std::vector<double>& worldInput)
{
    assert(worldInput.size() == topology_[INPUT]);

    // Assign world input values to input neurons
    for (int n = 0; n < worldInput.size(); ++n)
        net_[n].setOutput(worldInput[n]);

    // Forward propagate: activate every neuron
    // in every layer except input layer neurons
    for (int n = 0; n < net_.size(); ++n)
        net_[n].activate(net_, n);
}

void Ann::backProp(const std::vector<double>& target)
{
    assert(target.size() == topology_[topology_.size() - 1]);

    // Calculate overall net error (RMS of output neuron errors)
    int neuronsCount = 0;
    for (size_t i = 0; i < topology_.size(); ++i)
        neuronsCount += topology_[i] + 1;

    error_ = 0.0; // reset for each backpropagation
    int nout = neuronsCount - topology_[topology_.size() - 1] - 1;
    int ntar = 0;
    for (; nout < net_.size(); ++nout)
    {
        double delta = target[ntar++] - net_[nout].output();
        error_ += delta * delta;
    }

    error_ /= topology_[topology_.size() - 1]; // average
    error_ = sqrt(error_);                     // RMS

    // Implement a recent average measurement [for displaying, not related to learning]
    averageError_ = (averageError_ * averageSmoothingFactor_ + error_)
                                  / (averageSmoothingFactor_ + 1.0);

    // Calculate output layer gradients
    nout = neuronsCount - topology_[topology_.size() - 1] - 1;
    ntar = 0;
    for (; nout < net_.size(); ++nout)
        net_[nout].calcOutputGradients(target[ntar++]);

    // Calculate hidden layers gradients   
    for (size_t n = net_.size() - topology_[topology_.size() - 1] - 1; n > topology_[0]; --n)
        net_[n].calcHiddenGradients(net_);

    // Update connection weights
    // for all layers from output to first hidden layer
    for (size_t n = 0; n < net_.size() - topology_[topology_.size() - 1]; ++n)
        net_[n].updateInputWeights(net_);
}

void Ann::addNeuron(int layer, int neuron, bool isBias)
{
    assert(0 < layer && layer <= topology_.size());

    int neuronsCount = (layer == topology_.size()) ? topology_[layer - 1] : topology_[layer - 1] + 1;

    assert(0 < neuron && neuron <= neuronsCount);

    // Get input and output indexes of neurons for adding neuron
    std::vector<int> ins;
    std::vector<int> outs;

    int index = 0;
    for (int i = 0; i < layer - 1; ++i)
        index += topology_[i] + 1;

    index += isBias ? topology_[layer - 1] : neuron - 1;

    for (auto& n : net_)
    {
        n.updateInsOuts(index, true);
    }

    index -= isBias ? topology_[layer - 1] : neuron - 1;

    if (layer != 1 && !isBias)
    {
        for (int i = 0; i < topology_[layer - 2] + 1; ++i)
        {
            int nIndex = index - topology_[layer - 2] - 1 + i;
            ins.emplace_back(nIndex);
            net_[nIndex].addOut(neuron - 1);
        }
    }

    if (layer != topology_.size())
    {
        for (int i = 0; i < topology_[layer]; ++i)
        {
            int nIndex = index + topology_[layer - 1] + 1 + i;
            outs.emplace_back(nIndex);
            net_[nIndex].addIn();
        }
    }

    // Figure out position where neuron will be added
    auto it = net_.begin();
    for (int i = 0; i < index; ++i)
        ++it;

    net_.insert(it, Neuron(ins, outs, isBias));
    ++topology_[layer - 1];
}

void Ann::deleteNeuron(int layer, int neuron)
{
    assert(0 < layer && layer <= topology_.size());

    int neuronsCount = (layer == topology_.size()) ? topology_[layer - 1] : topology_[layer - 1] + 1;

    assert(0 < neuron && neuron <= neuronsCount);
   
    int index = 0;
    for (int i = 0; i < layer - 1; ++i)
        index += topology_[i] + 1;
    index += neuron - 1;

    net_[index].removeOuts(net_, index);
    net_[index].removeIns(net_, index);

    // Figure out position of neuron which will be erased
    auto it = net_.begin();
    for (int i = 0; i < index; ++i)
        ++it;

    net_.erase(it);
    --topology_[layer - 1];

    for (auto& n : net_)
    {
        n.updateInsOuts(index, false);
    }
}

void Ann::findIndexes(int srcLayer, int srcNeuron, int dstLayer, int dstNeuron, int& srcIndex, int& dstIndex)
{
    assert(0 < srcLayer && srcLayer <= topology_.size()
        && 0 < dstLayer && dstLayer <= topology_.size());

    int srcNeuronsCount = (srcLayer == topology_.size()) ? topology_[srcLayer - 1] : topology_[srcLayer - 1] + 1;
    int dstNeuronsCount = (dstLayer == topology_.size()) ? topology_[dstLayer - 1] : topology_[dstLayer - 1] + 1;

    assert( 0 < srcNeuron && srcNeuron <= srcNeuronsCount
         && 0 < dstNeuron && dstNeuron <= dstNeuronsCount);

    for (int i = 0; i < srcLayer - 1; ++i)
        srcIndex += topology_[i] + 1;

    for (int i = 0; i < dstLayer - 1; ++i)
        dstIndex += topology_[i] + 1;

    srcIndex += srcNeuron - 1;
    dstIndex += dstNeuron - 1;
}

void Ann::addConnection(int srcLayer, int srcNeuron, int dstLayer, int dstNeuron)
{
    int srcIndex = 0;
    int dstIndex = 0;
    findIndexes(srcLayer, srcNeuron, dstLayer, dstNeuron, srcIndex, dstIndex);

    net_[srcIndex].addConnection(dstIndex, false);
    net_[dstIndex].addConnection(srcIndex, true);
}

void Ann::deleteConnection(int srcLayer, int srcNeuron, int dstLayer, int dstNeuron)
{
    int srcIndex = 0;
    int dstIndex = 0;
    findIndexes(srcLayer, srcNeuron, dstLayer, dstNeuron, srcIndex, dstIndex);

    net_[srcIndex].deleteConnection(dstIndex, false);
    net_[dstIndex].deleteConnection(srcIndex, true);
}

std::vector<double> Ann::getOutput() const
{
    std::vector<double> result;

    int neuronsCount = 0;
    for (size_t i = 0; i < topology_.size(); ++i)
        neuronsCount += topology_[i] + 1;

    int n = neuronsCount - topology_[topology_.size() - 1] - 1;
    for (; n < net_.size(); ++n)
        result.emplace_back(net_[n].output());

    return result;
}

void Ann::reportState(const std::string& fileName)
{
    std::stringstream ss;
    for (int n = 0; n < net_.size(); ++n)
        net_[n].reportState();

    ss << "===================================="
       << std::endl;

    std::ofstream(fileName) << ss.rdbuf();
}
