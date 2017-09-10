#include "Net.h"
#include <iostream>
#include <cassert>
#include <fstream>

using namespace ANN;

double Ann::averageSmoothingFactor_ = 100.0; // Number of training samples to average over

Ann::Ann(const std::vector<int>& topology)
    : error_(0.0),
    averageError_(0.2),
    topology_(topology),
    neurons_(0)
{
    initializeNet();
}

void Ann::initializeNet()
{
    for (size_t i = 0; i < topology_.size(); ++i)
        neurons_ += topology_[i] + 1;
    --neurons_;

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
        for (; n < neurons_; ++n)
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

void Ann::trainNet(const std::vector<Sample>& trainSet, const double avrgError)
{
    while (averageError() > avrgError)
    {
        for (const auto& sample : trainSet)
        {
            pass++;
            feedForw(sample.input);
            backProp(sample.target);
        }
    }
}

void Ann::testNet(const std::vector<Sample>& trainSet)
{
    for (const auto& sample : trainSet)
    {
        feedForw(sample.input);

        display("Input: ", sample.input);
        display("Target:", sample.target);
        display("Output:", getOutput());
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
    error_ = 0.0; // reset for each backpropagation
    int nout = neurons_ - topology_[topology_.size() - 1];
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
    nout = neurons_ - topology_[topology_.size() - 1];
    ntar = 0;
    for (; nout < net_.size(); ++nout)
        net_[nout].calcOutputGradients(target[ntar++]);

    // Calculate hidden layers gradients   
    for (size_t n = net_.size() - topology_[topology_.size() - 1] - 1; n > 0; --n)
        net_[n].calcHiddenGradients(net_);

    // Update connection weights
    // for all layers from output to first hidden layer
    for (size_t n = 0; n < net_.size() - topology_[topology_.size() - 1]; ++n)
        net_[n].updateInputWeights(net_);
}

void Ann::addNeuron(const NC& nc, bool isBias)
{
    assert(0 < nc.layer && nc.layer <= topology_.size());

    int neurons = topology_[nc.layer - 1];
    neurons += (nc.layer == topology_.size()) ? 0 : 1;

    assert(0 < nc.neuron && nc.neuron <= neurons);

    // Get input and output indexes of neurons for adding neuron
    std::vector<int> ins;
    std::vector<int> outs;

    int index = 0;
    for (int i = 0; i < nc.layer - 1; ++i)
        index += topology_[i] + 1;

    index += isBias ? topology_[nc.layer - 1] : nc.neuron - 1;

    for (auto& n : net_)
    {
        n.updateInsOuts(index, true);
    }

    index -= isBias ? topology_[nc.layer - 1] : nc.neuron - 1;

    if (nc.layer != 1 && !isBias)
    {
        for (int i = 0; i < topology_[nc.layer - 2] + 1; ++i)
        {
            int nIndex = index - topology_[nc.layer - 2] - 1 + i;
            ins.emplace_back(nIndex);
            net_[nIndex].addOut(nc.neuron - 1);
        }
    }

    if (nc.layer != topology_.size())
    {
        for (int i = 0; i < topology_[nc.layer]; ++i)
        {
            int nIndex = index + topology_[nc.layer - 1] + 1 + i;
            outs.emplace_back(nIndex);
            net_[nIndex].addIn();
        }
    }

    // Figure out position where neuron will be added
    auto it = net_.begin();
    index += isBias ? topology_[nc.layer - 1] + 1 : nc.neuron - 1;
    std::advance(it, index);

    net_.insert(it, Neuron(ins, outs, isBias));
    ++topology_[nc.layer - 1];
}

void Ann::deleteNeuron(const NC& nc)
{
    assert(0 < nc.layer && nc.layer <= topology_.size());

    int neurons = topology_[nc.layer - 1];
    neurons += (nc.layer == topology_.size()) ? 0 : 1;

    assert(0 < nc.neuron && nc.neuron <= neurons);
   
    int index = 0;
    for (int i = 0; i < nc.layer - 1; ++i)
        index += topology_[i] + 1;
    index += nc.neuron - 1;

    net_[index].removeOuts(net_, index);
    net_[index].removeIns(net_, index);

    // Figure out position of neuron which will be erased
    auto it = net_.begin();
    std::advance(it, index);

    net_.erase(it);
    --topology_[nc.layer - 1];

    for (auto& n : net_)
    {
        n.updateInsOuts(index, false);
    }
}

void Ann::findIndexes(const NC& srcNC, const NC& dstNC, int& srcIndex, int& dstIndex)
{
    assert((0 < srcNC.layer && srcNC.layer <= topology_.size())
        && (0 < dstNC.layer && dstNC.layer <= topology_.size()));

    int srcNeurons = (srcNC.layer == topology_.size()) ? topology_[srcNC.layer - 1] : topology_[srcNC.layer - 1] + 1;
    int dstNeurons = (dstNC.layer == topology_.size()) ? topology_[dstNC.layer - 1] : topology_[dstNC.layer - 1] + 1;

    assert((0 < srcNC.neuron && srcNC.neuron <= srcNeurons)
        && (0 < dstNC.neuron && dstNC.neuron <= dstNeurons));

    for (int i = 0; i < srcNC.layer - 1; ++i)
        srcIndex += topology_[i] + 1;

    for (int i = 0; i < dstNC.layer - 1; ++i)
        dstIndex += topology_[i] + 1;

    srcIndex += srcNC.neuron - 1;
    dstIndex += dstNC.neuron - 1;
}

void Ann::addConnection(const NC& srcNC, const NC& dstNC)
{
    int srcIndex = 0;
    int dstIndex = 0;
    findIndexes(srcNC, dstNC, srcIndex, dstIndex);

    net_[srcIndex].addConnection(dstIndex, false);
    net_[dstIndex].addConnection(srcIndex, true);
}

void Ann::deleteConnection(const NC& srcNC, const NC& dstNC)
{
    int srcIndex = 0;
    int dstIndex = 0;
    findIndexes(srcNC, dstNC, srcIndex, dstIndex);

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
    int layer  = 0;
    int neuron = 0;
    for (auto& t : topology_)
    {
        ++layer;
        t += (layer == topology_.size()) ? 0 : 1;
        for (int n = 0; n < t; ++n)
        {
            ss << "=================="                      << std::endl;
            ss << "Layer_" << layer << "  Neuron_" << n + 1 << std::endl;
            ss << net_[neuron++].reportState()              << std::endl;
        }
    }
    ss << "==================";
    std::ofstream(fileName) << ss.rdbuf();
}
