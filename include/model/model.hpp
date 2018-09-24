#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <armadillo>

namespace sensei {
class model {
public:
	virtual void fit(const arma::mat&, const arma::mat&) = 0;
	virtual arma::mat predict(const arma::mat&) const = 0;
	virtual void summary() const = 0;
};
}

#endif // __MODEL_HPP__

