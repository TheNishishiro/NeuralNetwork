#ifndef NNLayer_H
#define NNLayer_H

#include <string>

using namespace std;
class NeuralNetworkLayer
{
private:
	string activ = "relu";
	const int minRandomWeightsValue = 0, maxRandomWeightsValue = 200;

	double ActivationFunction(string activation, double x);
	double DerivativeActivationFunction(string activation, int i);

public:
	int NumberOfNodes;
	int NumberOfChildNodes;
	int NumberOfParentNodes;
	double **Weights;
	double **WeightChanges;
	double *NeuronValues;
	double *DesiredValues;
	double *Errors;
	double *BiasWeights;
	double *BiasValues;
	double LearningRate;
	bool LinearOutput;
	bool UseMomentum;
	double MomentumFactor;
	NeuralNetworkLayer *ParentLayer;
	NeuralNetworkLayer *ChildLayer;


	NeuralNetworkLayer();
	~NeuralNetworkLayer();

	void Initialize(int NumNodes, NeuralNetworkLayer *parent, NeuralNetworkLayer *child);
	void RandomizeWeights();
	void CalculateNeuronValues();
	void CalculateErrors();
	void AdjustWeights();

};

#endif