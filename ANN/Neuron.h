#ifndef NEURON_H
#define NEURON_H

#include "General.h"
#include <vector>

namespace ANN
{
    class   Neuron;
    struct  Connection;
    typedef Vector<Neuron> Layer;

    class Neuron
    {
    public:
               Neuron(int numOutputs, int myIdxL);
               void   connect(const Neuron& to);
               void   activate(const Layer& prevLayer);
               void   calcOutputGradients(const double target);
               void   calcHiddenGradients(const Layer& nextLayer);
               void   updateInputWeights(Layer& prevLayer);
               void   setOutput(double val) { output_ = val;  }
               double output() const        { return output_; }

    private:
        static double activationFunction(double x);           // for forward  propagation
        static double activationFunctionDerivative(double x); // for backward propagation

               double sumDOW(const Layer& nextLayer) const;

    // TODO: Move to another module.
    public:

        static double rate;     // [0.0, 1.0] overall net learning rate
        static double momentum; // [0.0,   n] multiplier of last weight change (momentum)
               double output_;    double outputDup_;
               double gradient_;  double gradientDup_;
        const  int    idxL_;

        Vector<Connection> axon_;

        /////////////////////// UC ///////////////////////
        Vector<Connection> inCons_;
        Vector<Connection> axCons_;
    };

    struct Connection
    {
        /////////////////////// UC ///////////////////////
        const Neuron* link = nullptr;
        double value;

        double weight = rand() / double(RAND_MAX);
        double deltaWeight;
    };
} // namespace ANN

#endif // NEURON_H
