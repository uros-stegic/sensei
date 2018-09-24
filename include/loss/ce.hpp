#ifndef __CE_HPP__
#define __CE_HPP__

#include <armadillo>
#include <loss/loss.hpp>

namespace sensei {
struct ce : public loss {
	double evaluate(const arma::mat&, const arma::mat&) const override;
	arma::mat gradient(const arma::mat&, const arma::mat&) const override;
	void summary() const override;
};
}

#endif // __CE_HPP__

