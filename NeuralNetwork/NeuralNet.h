#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include "NeuralNetworkLayer.h"
using namespace std;
class NeuralNet
{
private:
	int numberOfLayers;
public:
	vector<NeuralNetworkLayer*> Layers;


	NeuralNet(int nOfLayers, int *nNodesInLayers);
	~NeuralNet();

	void SetInput(int i, double value);
	double GetOutput(int i);
	void SetDesiredOutput(int i, double value);
	void FeedForward(); 
	void BackPropagate();
	int GetMaxOutputID();
	double CalculateError();
	void SetLearningRate(double rate);
	void SetLinearOutput(bool useLinear);
	void SetMomentum(bool useMomentum, double factor);
};

#endif
