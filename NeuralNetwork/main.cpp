
#include "stdafx.h"
#include "NeuralNet.h"
#include <time.h>
#include <iostream>

using namespace std;

int main()
{
	srand(time(NULL));
	int nodesInLayers[] = { 2, 5, 6, 2 };
	int numberOfLayers = 4;

	NeuralNet NN(numberOfLayers, nodesInLayers);
	NN.SetLearningRate(0.01);
	NN.SetLinearOutput(false);
	NN.SetMomentum(true, 0.9);

	double GlobalcorrectRate = 0, GlobalcorrectRatio = 0;
	double c = 0, epochs = 0;

	// Train network
	while (epochs < 6)
	{
		c++;
		double x = 0, y = 0, error = 0, nodeActive = 0;
		double output[2] = { 0,0 };

		x = static_cast<double>(rand() % 1300);
		y = static_cast<double>(rand() % 700);

		NN.SetInput(0, x / 1300);
		NN.SetInput(1, y / 700);

		if (y < 350)
		{
			output[0] = 1;
			nodeActive = 0;
		}
		else
		{
			output[1] = 1;
			nodeActive = 1;
		}

		for (int i = 0; i < nodesInLayers[numberOfLayers - 1]; i++)
		{
			NN.SetDesiredOutput(i, output[i]);
		}

		NN.FeedForward();
		error += NN.CalculateError();
		if (nodeActive == NN.GetMaxOutputID())
			GlobalcorrectRate++;

		
		
		GlobalcorrectRatio = GlobalcorrectRate / c;
		NN.BackPropagate();

		if (c > 1000)
		{
			cout << "correct rate in epoch " << epochs + 1 << " is " << GlobalcorrectRatio << endl;
			GlobalcorrectRate = 0;
			c = 0;
			epochs++;
		}
	}

	//Test
	GlobalcorrectRate = 0;
	GlobalcorrectRatio = 0;
	c = 0;
	while (c < 500)
	{
		c++;
		double x = 0, y = 0, error = 0, nodeActive = 0;
		double output[2] = { 0,0 };

		x = static_cast<double>(rand() % 1300);
		y = static_cast<double>(rand() % 700);

		NN.SetInput(0, x / 1300);
		NN.SetInput(1, y / 700);

		if (y < 350)
		{
			output[0] = 1;
			nodeActive = 0;
		}
		else
		{
			output[1] = 1;
			nodeActive = 1;
		}

		for (int i = 0; i < nodesInLayers[numberOfLayers - 1]; i++)
		{
			NN.SetDesiredOutput(i, output[i]);
		}

		NN.FeedForward();
		error += NN.CalculateError();
		if (nodeActive == NN.GetMaxOutputID())
			GlobalcorrectRate++;

		GlobalcorrectRatio = GlobalcorrectRate / c;
	}

	cout << "Correct rate from test: " << GlobalcorrectRatio << endl;

	system("pause");
    return 0;
}

