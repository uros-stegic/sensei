#ifndef __LINEAR_HPP__
#define __LINEAR_HPP__

#include <armadillo>
#include <activations/activation.hpp>

namespace sensei {
struct linear : public activation {
	arma::mat evaluate(const arma::mat&) const override;
	arma::mat gradient(const arma::mat&) const override;
	void summary() const override;
};
}

#endif // __LINEAR_HPP__

