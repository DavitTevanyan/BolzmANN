#ifndef  NET_H
#define  NET_H

#include "Neuron.h"
#include <fstream>

namespace ANN {

    class Ann
    {
    public:
        enum { INPUT };
        explicit Ann(const std::vector<int>& topology);      
        void                feedForw(const std::vector<double>& worldInput);
        void                backProp(const std::vector<double>& target);

        std::vector<double> getOutput()    const;
        double              averageError() const { return averageError_; }
        void                dumpNN();
    private:
        std::fstream       outFile_;
        double             error_;
        double             averageError_;
        std::vector<Layer> layers_;                       // layers_[layer][neuron]
        static double      averageSmoothingFactor_;
    };

} // namespace ANN

#endif // NET_H
