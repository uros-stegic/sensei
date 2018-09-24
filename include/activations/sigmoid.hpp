#ifndef __SIGMOID_HPP__
#define __SIGMOID_HPP__

#include <armadillo>
#include <activations/activation.hpp>

namespace sensei {
struct sigmoid : public activation {
	arma::mat evaluate(const arma::mat&) const override;
	arma::mat gradient(const arma::mat&) const override;
	void summary() const override;
};
}

#endif // __SIGMOID_HPP__

