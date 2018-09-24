#ifndef __TANH_HPP__
#define __TANH_HPP__

#include <armadillo>
#include <activations/activation.hpp>

namespace sensei {
struct tanh : public activation {
	arma::mat evaluate(const arma::mat&) const override;
	arma::mat gradient(const arma::mat&) const override;
	void summary() const override;
};
}

#endif // __TANH_HPP__

