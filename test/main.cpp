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
	auto dataset = load_csv("../test/dataset/big.csv");
	arma::mat X = dataset.cols(0, 2).t();
	arma::mat y = dataset.col(3).t();

	arma::mat y0 = arma::zeros(2, y.n_cols);
	y0.row(0) = 1-y;
	y0.row(1) = y;

	auto optimizer = new gradient_descent(
		new mse,	// loss
		0.1,		// learning rate
		100u		// #epochs
	);

	model *my_first_model = new rnn(3, {
		make_cell<sensei::tanh, 10>(.5),
		make_cell<sensei::tanh, 20>(.5),
		make_cell<softmax, 2>()
	}, optimizer);

	my_first_model->fit(X, y0);

	std::cout << "ground truth: " << std::endl;
	std::cout << y0.cols(1, 3) << std::endl;

	std::cout << "predictions: " << std::endl;
	std::cout << my_first_model->predict(X.cols(1,3)) << std::endl;

	return 0;
}

