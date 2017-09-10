#pragma once

#include "SpecificUtilities.h"
#include "TrainingData.h"
#include "Neuron.h"

namespace ANN {

    class Ann
    {
    public:
        enum { INPUT };
        explicit  Ann(const std::vector<int>& topology);
        void trainNet(const std::vector<Sample>& trainSet, const double avrgError);
        void testNet( const std::vector<Sample>& trainSet);

        void addNeuron(       const NC& nc, bool isBias);
        void deleteNeuron(    const NC& nc);
        void addConnection(   const NC& srcNC, const NC& dstNC);
        void deleteConnection(const NC& srcNC, const NC& dstNC);

        std::vector<double> getOutput() const;
        double              averageError() const { return averageError_; }
        void                reportState(const std::string& fileName);

    private:
        void findIndexes(const NC& srcNC, const NC& dstNC, int& srcIndex, int& dstIndex);
        void initializeNet();
        void feedForw(const std::vector<double>& worldInput);
        void backProp(const std::vector<double>& target);

    private:
        double              error_;
        double              averageError_;
        std::vector<Neuron> net_;
        std::vector<int>    topology_;
        int                 neurons_;
        static double       averageSmoothingFactor_;
    };

} // namespace ANN