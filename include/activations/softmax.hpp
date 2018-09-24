#ifndef __SOFTMAX_HPP__
#define __SOFTMAX_HPP__

#include <armadillo>
#include <activations/activation.hpp>

namespace sensei {
struct softmax : public activation {
	arma::mat evaluate(const arma::mat&) const override;
	arma::mat gradient(const arma::mat&) const override;
	void summary() const override;
};
}

#endif // __SOFTMAX_HPP__

