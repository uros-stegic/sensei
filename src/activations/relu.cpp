#include <activations/relu.hpp>

using namespace sensei;

arma::mat relu::evaluate(const arma::mat& input) const
{
	return arma::clamp(input, 0, input.max());
}
arma::mat relu::gradient(const arma::mat& input) const
{
	arma::mat norms = arma::sign(input);
	return evaluate(input);
}
void relu::summary() const
{
	std::cout << "activation: relu" << std::endl;
}

