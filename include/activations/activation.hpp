#ifndef __ACTIVATION_HPP__
#define __ACTIVATION_HPP__

#include <armadillo>

namespace sensei {
struct activation {
	virtual ~activation() {}
	virtual arma::mat evaluate(const arma::mat&) const = 0;
	virtual arma::mat gradient(const arma::mat&) const = 0;
	virtual void summary() const = 0;
};
}

#endif // __ACTIVATION_HPP__

