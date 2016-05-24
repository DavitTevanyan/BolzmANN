#ifndef  NET_H
#define  NET_H

#include "Neuron.h"

namespace ANN
{
    class Net
    {
    public:
        explicit Net(const std::vector<int>& topology);      
        std::vector<double> getResults() const;
        double              getRecentAverageError() const { return recentAverageError_; }
        void                feedForward(const std::vector<double>& inputVals);
        void                backProp(const std::vector<double>& targetVals);

    private:
        double             error_;
        double             recentAverageError_;
        std::vector<Layer> layers_;                       // layers_[layerNum][neuronNum]
        static double      recentAverageSmoothingFactor_;
    };
}

#endif // NET_H
