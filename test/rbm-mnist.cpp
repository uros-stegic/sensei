#include <iostream>
#include <fstream>
#include <model/rnn.hpp>
#include <model/rbm.hpp>
#include <optimizer/gradient_descent.hpp>
#include <activations/sigmoid.hpp>
#include <activations/tanh.hpp>
#include <activations/softmax.hpp>
#include <loss/mse.hpp>
#include <loss/ce.hpp>
#include <data/data_io.hpp>
#include <armadillo>
#include <utils/armadillo_utils.hpp>

using namespace sensei;

int main(int argc, char** argv)
{
	auto dataset = load_csv("../test/dataset/mnist.csv");
	std::cout << shape(dataset) << std::endl;

	rbm model(784, 100);
	model.fit(dataset, 0.1, 10);

	return 0;
}

