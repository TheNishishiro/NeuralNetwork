#include "stdafx.h"
#include "NeuralNetworkLayer.h"
#include <string>
#include <time.h>
#include <math.h>

using namespace std;
NeuralNetworkLayer::NeuralNetworkLayer()
{
	ParentLayer = NULL;
	ChildLayer = NULL;
	LinearOutput = false;
	UseMomentum = false;
	MomentumFactor = 0.9;
}


NeuralNetworkLayer::~NeuralNetworkLayer()
{
	cout << "NeuralNetworkLayer destructor" << endl;
	if (ChildLayer != NULL)
	{
		for (int i = 0; i < NumberOfNodes; i++)
		{
			delete[] Weights[i];
			delete[] WeightChanges[i];
		}
		delete[] Weights;
		delete[] WeightChanges;
	}
	delete[] NeuronValues;
	delete[] DesiredValues;
	delete[] Errors;
	delete[] BiasValues;
	delete[] BiasWeights;
}

void NeuralNetworkLayer::Initialize(int NumNodes, NeuralNetworkLayer *parent, NeuralNetworkLayer *child)
{
	NeuronValues = new double[NumberOfNodes];
	DesiredValues = new double[NumberOfNodes];
	Errors = new double[NumberOfNodes];

	if (parent != NULL)
		ParentLayer = parent;
	if (child != NULL)
	{
		ChildLayer = child;
		Weights = new double*[NumberOfNodes];
		WeightChanges = new double*[NumberOfNodes];

		for (int i = 0; i < NumberOfNodes; i++)
		{
			Weights[i] = new double[NumberOfChildNodes];
			WeightChanges[i] = new double[NumberOfChildNodes];
		}
		BiasValues = new double[NumberOfChildNodes];
		BiasWeights = new double[NumberOfChildNodes];
	}
	else
	{
		Weights = NULL;
		BiasValues = NULL;
		BiasWeights = NULL;
		WeightChanges = NULL;
	}

	for (int i = 0; i < NumberOfNodes; i++)
	{
		NeuronValues[i] = 0;
		DesiredValues[i] = 0;
		Errors[i] = 0;

		if (ChildLayer != NULL)
		{
			for (int j = 0; j < NumberOfChildNodes; j++)
			{
				Weights[i][j] = 0;
				WeightChanges[i][j] = 0;
			}
		}
	}

	if (ChildLayer != NULL)
	{
		for (int i = 0; i < NumberOfChildNodes; i++)
		{
			BiasValues[i] = 1;
			BiasWeights[i] = 0;
		}
	}
}
void NeuralNetworkLayer::RandomizeWeights() 
{
	for (int i = 0; i < NumberOfNodes; i++)
		for (int j = 0; j < NumberOfChildNodes; j++)
			Weights[i][j] = (static_cast<double>(rand() % maxRandomWeightsValue + minRandomWeightsValue) / 100) - 1;
	for(int i = 0; i < NumberOfChildNodes; i++)
		BiasWeights[i] = (static_cast<double>(rand() % maxRandomWeightsValue + minRandomWeightsValue) / 100) - 1;

}
void NeuralNetworkLayer::CalculateNeuronValues()
{
	if (ParentLayer != NULL)
	{
		for (int j = 0; j < NumberOfNodes; j++)
		{
			double sumOfWieghtedPreviousNeuronsValues = 0;
			for (int i = 0; i < NumberOfParentNodes; i++)
				sumOfWieghtedPreviousNeuronsValues += ParentLayer->NeuronValues[i] * ParentLayer->Weights[i][j];
			sumOfWieghtedPreviousNeuronsValues += ParentLayer->BiasValues[j] * ParentLayer->BiasWeights[j];
			if (ChildLayer == NULL && LinearOutput)
				NeuronValues[j] = sumOfWieghtedPreviousNeuronsValues;
			else
				NeuronValues[j] = ActivationFunction(activ, sumOfWieghtedPreviousNeuronsValues);
		}
	}
}
void NeuralNetworkLayer::CalculateErrors()
{
	double sum, derivative = 0.01;

	if (ChildLayer == NULL) // If we are in the output layer calculate error of our output
	{
		for (int i = 0; i < NumberOfNodes; i++)
		{
			derivative = DerivativeActivationFunction(activ, i);
			Errors[i] = (DesiredValues[i] - NeuronValues[i]) * derivative;
		}
	}
	else if (ParentLayer == NULL) // If we are in our input layer there is no need for error
	{
		for (int i = 0; i < NumberOfNodes; i++)
		{
			Errors[i] = 0;
		}
	}
	else // if we are in hidden layer propagate error from child layer and scale it with weight
	{
		for (int i = 0; i < NumberOfNodes; i++)
		{
			sum = 0;
			for (int j = 0; j < NumberOfChildNodes; j++)
			{
				sum += ChildLayer->Errors[j] * Weights[i][j];
			}
			derivative = DerivativeActivationFunction(activ, i);
			Errors[i] = sum * derivative;
		}
	}
}
void NeuralNetworkLayer::AdjustWeights()
{
	double dw;

	if (ChildLayer != NULL)
	{
		for (int i = 0; i < NumberOfNodes; i++)
		{
			for (int j = 0; j < NumberOfChildNodes; j++)
			{
				dw = LearningRate * ChildLayer->Errors[j] * NeuronValues[i];
				if (UseMomentum)
				{
					Weights[i][j] += dw + MomentumFactor * WeightChanges[i][j];
					WeightChanges[i][j] = dw;
				}
				else
				{
					Weights[i][j] += dw;
				}
			}
		}

		for (int i = 0; i < NumberOfChildNodes; i++)
		{
			BiasWeights[i] += LearningRate * ChildLayer->Errors[i] * BiasValues[i];
		}
	}
}
double NeuralNetworkLayer::ActivationFunction(string activation, double x)
{
	if (activation == "sigmoid")
		return 1.0 / (1 + exp(-x));
	else if (activation == "tanh")
		return tanh(x);
	else if (activation == "relu")
	{
		if (x < 0)
			return x * 0.01;
		else
			return x;
	}

	return 0;
}
double NeuralNetworkLayer::DerivativeActivationFunction(string activation, int i)
{
	if (activation == "sigmoid")
		return NeuronValues[i] * (1 - NeuronValues[i]);
	else if (activation == "tanh")
		return pow(NeuronValues[i], 2);
	else if (activation == "relu")
	{
		if (NeuronValues[i] < 0)
			return 0.01;
		else
			return 1;
	}

	return 0;
}

