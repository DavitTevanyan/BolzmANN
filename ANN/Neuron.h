#pragma once

#include "General.h"

namespace ANN
{
    class  Neuron;
    struct Connection;
    using  Layer = Vector<Neuron>;

    class Neuron
    {
    public:
        explicit Neuron(const std::vector<int>& ins, const std::vector<int>& outs, bool isBias);
        void   activate(const std::vector<Neuron>& net, int n);
        void   calcOutputGradients(const double target);
        void   calcHiddenGradients(const std::vector<Neuron>& net);
        void   updateInputWeights(const std::vector<Neuron>& net);
        void   setOutput(double val) { output_ = val;  }
        double output() const        { return output_; }
        void   addIn();
        void   addOut(int pos);
        void   updateInsOuts(int index, bool operation);
        void   removeIns(std::vector<Neuron>& net, int index);
        void   removeOuts(std::vector<Neuron>& net, int index);
        void   addConnection(int index, bool direction);
        void   deleteConnection(int index, bool direction);
        std::string reportState();

    private:
        static double af(double x);            // for forward  propagation
        static double af_Derivative(double x); // for backward propagation

    private:
        static double           rate;     // [0.0, 1.0] overall net learning rate
        static double           momentum; // [0.0,   n] multiplier of last weight change (momentum)
               double           output_;
               double           gradient_;
               std::vector<int> ins_;
               std::vector<int> outs_;
               bool             isBias_;
        Vector<Connection>      axon_;
    };

    struct Connection
    {
        double weight = rand() / double(RAND_MAX);
        double deltaWeight;
    };
} // namespace ANN