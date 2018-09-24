#include <loss/mse.hpp>

using namespace sensei;

double mse::evaluate(const arma::mat& target, const arma::mat& preds) const
{
	return 0.5 * arma::accu(arma::mean(arma::sum(arma::pow(target - preds, 2)), 1));
}
arma::mat mse::gradient(const arma::mat& target, const arma::mat& preds) const
{
	return target - preds;
}
void mse::summary() const
{
	std::cout << "loss: MSE" << std::endl;
}

