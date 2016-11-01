#ifndef NEURON_H
#define NEURON_H

#include <vector>

namespace ANN
{
	struct Connection
	{
		double weight;
		double deltaWeight;
	};

	class Neuron;
	typedef std::vector<Neuron> Layer;

	class Neuron
	{
	public:
		Neuron(int numOutputs, int myId);
		void   setOutputVal(const double val)        { outputVal_ = val; }
		double outputVal() const                     { return outputVal_; }
		void   feedForward(const Layer& prevLayer);
		void   calcOutputGradients(const double targetVal);
		void   calcHiddenGradients(const Layer& nextLayer);
		void   updateInputWeights(Layer& prevLayer);

	private:
		static double transferFunction(const double x);
		static double transferFunctionDerivative(const double x);
		static double randomWeight()                            { return rand() / double(RAND_MAX); }
		       double sumDOW(const Layer& nextLayer) const;

        static double           eta;             // [0.0..1.0] overall net training rate
        static double           alpha;           // [0.0..n] multiplier of last weight change (momentum)
        double                  outputVal_;
		double                  gradient_;
        int                     id_;
        std::vector<Connection> connections_;
	};
}

#endif // NEURON_H
