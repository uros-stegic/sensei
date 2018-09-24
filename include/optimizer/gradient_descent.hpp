#ifndef __GRADIENT_DESCENT_HPP__
#define __GRADIENT_DESCENT_HPP__

#include <armadillo>
#include <memory>
#include <vector>

#include <model/rnn.hpp>
#include <loss/loss.hpp>
#include <optimizer/optimizer.hpp>

namespace sensei {
class gradient_descent : public optimizer {
public:
	gradient_descent(loss*, double, unsigned int);
	void optimize(rnn*, const arma::mat&, const arma::mat&) override;
	void summary() const;
};
}

#endif // __GRADIENT_DESCENT_HPP__

