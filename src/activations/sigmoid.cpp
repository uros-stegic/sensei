#include <activations/sigmoid.hpp>

using namespace sensei;

arma::mat sigmoid::evaluate(const arma::mat& input) const
{
	return arma::pow(1+arma::exp(-input), -1);
}
arma::mat sigmoid::gradient(const arma::mat& input) const
{
	arma::mat tmp = evaluate(input);
	return tmp % (1 - tmp);
}
void sigmoid::summary() const
{
	std::cout << "activation: sigmoid" << std::endl;
}

