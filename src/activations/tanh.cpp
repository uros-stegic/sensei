#include <activations/tanh.hpp>

using namespace sensei;

arma::mat tanh::evaluate(const arma::mat& input) const
{
	return arma::tanh(input);
}
arma::mat tanh::gradient(const arma::mat& input) const
{
	return 1 - arma::pow(evaluate(input), 2);
}
void tanh::summary() const
{
	std::cout << "activation: tanh" << std::endl;
}

