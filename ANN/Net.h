#ifndef  NET_H
#define  NET_H

#include "Neuron.h"

namespace ANN
{
    class Net
    {
    public:
        explicit Net(const std::vector<int>& topology);      
        void                feedForw(const std::vector<double>& worldInput);
        void                backProp(const std::vector<double>& target);

        std::vector<double> getResult() const;
        double              getRecentAverageError() const { return recentAverageError_; }

    private:
        double             error_;
        double             recentAverageError_;
        std::vector<Layer> layers_;                       // layers_[layerNum][neuronNum]
        static double      recentAverageSmoothingFactor_;
    };
}

#endif // NET_H
