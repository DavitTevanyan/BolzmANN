#pragma once

#include "Neuron.h"

namespace ANN {

    class Ann
    {
    public:
        enum { INPUT };
        explicit  Ann(const std::vector<int>& topology);
        void feedForw(const std::vector<double>& worldInput);
        void backProp(const std::vector<double>& target);
        void addNeuron(int layer, int neuron, bool isBias);
        void deleteNeuron(int layer, int neuron);
        void addConnection(int srcLayer, int srcNeuron, int dstLayer, int dstNeuron);
        void deleteConnection(int srcLayer, int srcNeuron, int dstLayer, int dstNeuron);

        std::vector<double> getOutput() const;
        double              averageError() const { return averageError_; }
        void                reportState(const std::string& fileName);

    private:
        void findIndexes(int srcLayer, int srcNeuron, int dstLayer, int dstNeuron, int& srcIndex, int& dstIndex);
        void initializeNet();
    private:
        double              error_;
        double              averageError_;
        std::vector<Neuron> net_;
        std::vector<int>    topology_;
        static double       averageSmoothingFactor_;
    };

} // namespace ANN