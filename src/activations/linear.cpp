#include <activations/linear.hpp>

using namespace sensei;

arma::mat linear::evaluate(const arma::mat& input) const
{
	return input;
}
arma::mat linear::gradient(const arma::mat& input) const
{
	return arma::ones(input.n_cols);
}
void linear::summary() const
{
	std::cout << "activation: linear" << std::endl;
}

