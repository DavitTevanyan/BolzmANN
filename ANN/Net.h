#ifndef  NET_H
#define  NET_H

#include "Neuron.h"

namespace ANN {

    class Ann
    {
    public:
        enum { INPUT };
        explicit  Ann(const std::vector<int>& topology);
        void feedForw(const std::vector<double>& worldInput);
        void backProp(const std::vector<double>& target);

        std::vector<double> getOutput()    const;
        double              averageError() const { return averageError_; }
        void                reportState(const std::string& fileName);

    private:
        double              error_;
        double              averageError_;
        std::vector<Neuron> net_;
        std::vector<int>    topology_;
        static double       averageSmoothingFactor_;
    };

} // namespace ANN

#endif // NET_H
