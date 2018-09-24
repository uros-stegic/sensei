#ifndef __MSE_HPP__
#define __MSE_HPP__

#include <armadillo>
#include <loss/loss.hpp>

namespace sensei {
struct mse : public loss {
	double evaluate(const arma::mat&, const arma::mat&) const override;
	arma::mat gradient(const arma::mat&, const arma::mat&) const override;
	void summary() const override;
};
}

#endif // __MSE_HPP__

