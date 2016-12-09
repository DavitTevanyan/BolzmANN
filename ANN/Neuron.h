#ifndef NEURON_H
#define NEURON_H

#include "General.h"

namespace ANN
{
    class   Neuron;
    struct  Connection;
    typedef Vector<Neuron> Layer;

    class Neuron
    {
    public:
        explicit Neuron(const std::vector<int>& in, const std::vector<int>& out, bool isBias);
        void   activate(const std::vector<Neuron>& net, int n);
        void   calcOutputGradients(const double target);
        void   calcHiddenGradients(const std::vector<Neuron>& net);
        void   updateInputWeights(const std::vector<Neuron>& net);
        void   setOutput(double val) { output_ = val;  }
        double output() const        { return output_; }

        std::string reportState();

    private:
        static double af(double x);            // for forward  propagation
        static double af_Derivative(double x); // for backward propagation

    private:
        static double           rate;     // [0.0, 1.0] overall net learning rate
        static double           momentum; // [0.0,   n] multiplier of last weight change (momentum)
               double           output_;
               double           gradient_;
               std::vector<int> in_;
               std::vector<int> out_;
               bool             isBias_;
        Vector<Connection>      axon_;
    };

    struct Connection
    {
        double weight = rand() / double(RAND_MAX);
        double deltaWeight;
    };
} // namespace ANN

#endif // NEURON_H
