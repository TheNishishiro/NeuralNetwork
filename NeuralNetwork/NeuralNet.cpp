#include "stdafx.h"
#include "NeuralNet.h"

using namespace std;

NeuralNet::NeuralNet(int nOfLayers, int *nNodesInLayers)
{
	numberOfLayers = nOfLayers;
	for (int i = 0; i < nOfLayers; i++)
		Layers.push_back(new NeuralNetworkLayer());
	for (int i = 0; i < nOfLayers; i++)
	{
		Layers[i]->NumberOfNodes = nNodesInLayers[i];
		if (i == 0)
		{
			Layers[i]->NumberOfChildNodes = nNodesInLayers[i + 1];
			Layers[i]->NumberOfParentNodes = 0;
			Layers[i]->Initialize(nNodesInLayers[i], NULL, Layers[i + 1]);
			Layers[i]->RandomizeWeights();
		}
		else if (i == nOfLayers - 1) 
		{
			Layers[i]->NumberOfChildNodes = 0;
			Layers[i]->NumberOfParentNodes = nNodesInLayers[i - 1];
			Layers[i]->Initialize(nNodesInLayers[i], Layers[i - 1], NULL);
		}
		else
		{
			Layers[i]->NumberOfChildNodes = nNodesInLayers[i + 1];
			Layers[i]->NumberOfParentNodes = nNodesInLayers[i - 1];
			Layers[i]->Initialize(nNodesInLayers[i], Layers[i - 1], Layers[i + 1]);
			Layers[i]->RandomizeWeights();
		}
	}
}


NeuralNet::~NeuralNet()
{
}

void NeuralNet::SetInput(int i, double value)
{
	if (i >= 0 && i < Layers[0]->NumberOfNodes)
		Layers[0]->NeuronValues[i] = value;
}
double NeuralNet::GetOutput(int i)
{
	if (i >= 0 && Layers[numberOfLayers - 1]->NumberOfNodes)
		return Layers[numberOfLayers - 1]->NeuronValues[i];
	return INT_MAX; // Error code
}
void NeuralNet::SetDesiredOutput(int i, double value)
{
	if (i >= 0 && Layers[numberOfLayers - 1]->NumberOfNodes)
		Layers[numberOfLayers - 1]->DesiredValues[i] = value;
}
void NeuralNet::FeedForward()
{
	for (int i = 0; i < numberOfLayers; i++)
	{
		Layers[i]->CalculateNeuronValues();
	}
}
void NeuralNet::BackPropagate()
{
	for (int i = numberOfLayers - 1; i > 0; i--)
	{
		Layers[i]->CalculateErrors();
		Layers[i]->AdjustWeights();
	}
	Layers[0]->AdjustWeights();
}
int NeuralNet::GetMaxOutputID()
{
	int id = 0;
	double maxval;
	maxval = Layers[numberOfLayers - 1]->NeuronValues[0];
	for (int i = 1; i < Layers[numberOfLayers - 1]->NumberOfNodes; i++)
	{
		if (Layers[numberOfLayers - 1]->NeuronValues[i] > maxval)
		{
			maxval = Layers[numberOfLayers - 1]->NeuronValues[i];
			id = i;
		}
	}
	return id;
}
double NeuralNet::CalculateError()
{
	int i;
	double error = 0;
	for (i = 0; i < Layers[numberOfLayers - 1]->NumberOfNodes; i++)
	{
		error += pow(Layers[numberOfLayers - 1]->NeuronValues[i] - Layers[numberOfLayers - 1]->DesiredValues[i], 2);
	}
	error = error / Layers[numberOfLayers - 1]->NumberOfNodes;
	return error;
}
void NeuralNet::SetLearningRate(double rate)
{
	for (int i = 0; i < numberOfLayers; i++)
	{
		Layers[i]->LearningRate = rate;
	}
}
void NeuralNet::SetLinearOutput(bool useLinear)
{
	for (int i = 0; i < numberOfLayers; i++)
	{
		Layers[i]->LinearOutput = useLinear;
	}
}
void NeuralNet::SetMomentum(bool useMomentum, double factor)
{
	for (int i = 0; i < numberOfLayers; i++)
	{
		Layers[i]->UseMomentum = useMomentum;
		Layers[i]->MomentumFactor = factor;
	}
}