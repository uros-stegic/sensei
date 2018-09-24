#include <loss/ce.hpp>
#include <utils/armadillo_utils.hpp>

using namespace sensei;

double ce::evaluate(const arma::mat& target, const arma::mat& preds) const
{
	std::cout << arma::log(preds) << std::endl;
	return arma::accu(arma::mean(-arma::sum(target % arma::log(preds)), 1));
}
arma::mat ce::gradient(const arma::mat& target, const arma::mat& preds) const
{
	return -target - preds;
}
void ce::summary() const
{
	std::cout << "loss: cross entropy" << std::endl;
}

