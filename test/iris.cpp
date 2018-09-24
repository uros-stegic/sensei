#include <iostream>
#include <fstream>
#include <model/rnn.hpp>
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
	auto dataset = load_csv("dataset/iris.csv");
	arma::mat X = dataset.cols(0, 3).t();
	arma::mat y = dataset.cols(4, 6).t();

	auto optimizer = new gradient_descent(
		new mse,	// loss
		0.01,		// learning rate
		100u		// #epochs
	);

	model *my_first_model = new rnn(4, {
		make_cell<sensei::sigmoid, 10>(.5),
		make_cell<softmax, 3>()
	}, optimizer);

	my_first_model->fit(X, y);

	std::cout << "ground truth: " << std::endl;
	std::cout << y.cols(1, 3) << std::endl;

	std::cout << "predictions: " << std::endl;
	std::cout << my_first_model->predict(X.cols(1,3)) << std::endl;

	return 0;
}

