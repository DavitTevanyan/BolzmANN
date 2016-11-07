#ifndef  NET_H
#define  NET_H

#include "Neuron.h"

namespace ANN
{
    class Net
    {
    public:
        enum { INPUT };
        explicit Net(const std::vector<int>& topology);      
        void                feedForw(const std::vector<double>& worldInput);
        void                backProp(const std::vector<double>& target);

        std::vector<double> getResult() const;
        double              averageError() const { return averageError_; }

    private:
        double             error_;
        double             averageError_;
        std::vector<Layer> layers_;                       // layers_[layerNum][neuronNum]
        static double      averageSmoothingFactor_;
    };
}

#endif // NET_H
