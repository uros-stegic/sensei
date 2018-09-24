#ifndef __RELU_HPP__
#define __RELU_HPP__

#include <armadillo>
#include <activations/activation.hpp>

namespace sensei {
struct relu : public activation {
	arma::mat evaluate(const arma::mat&) const override;
	arma::mat gradient(const arma::mat&) const override;
	void summary() const override;
};
}

#endif // __RELU_HPP__

