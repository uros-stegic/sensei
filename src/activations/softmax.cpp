#include <activations/softmax.hpp>
#include <iostream>
#include <utils/armadillo_utils.hpp>

using namespace sensei;

arma::mat softmax::evaluate(const arma::mat& input) const
{
	arma::mat shift = arma::repmat(-arma::max(input, 0), input.n_rows, 1);
	arma::mat exps = arma::exp(input+shift);
	arma::mat sum = arma::repmat(arma::sum(exps), input.n_rows, 1);

	return exps/sum;
}
arma::mat softmax::gradient(const arma::mat& input) const
{
	auto tmp = evaluate(input);
	return tmp % (1 - tmp);
}
void softmax::summary() const
{
	std::cout << "activation: softmax" << std::endl;
}

