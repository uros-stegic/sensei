#ifndef __LOSS_HPP__
#define __LOSS_HPP__

#include <armadillo>

namespace sensei {
struct loss {
	virtual double evaluate(const arma::mat&, const arma::mat&) const = 0;
	virtual arma::mat gradient(const arma::mat&, const arma::mat&) const = 0;
	virtual void summary() const = 0;
};
}

#endif // __LOSS_HPP__

