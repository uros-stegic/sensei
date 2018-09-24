#ifndef __OPTIMIZER_HPP__
#define __OPTIMIZER_HPP__

#include <armadillo>
#include <memory>
#include <vector>

#include <model/rnn.hpp>
#include <loss/loss.hpp>

namespace sensei {
class optimizer {
public:
	optimizer(loss*, double, unsigned int);
	std::vector<double> loss_history() const;

	virtual void optimize(rnn*, const arma::mat&, const arma::mat&)= 0;
	virtual void summary() const = 0;

protected:
	std::shared_ptr<loss> m_loss;
	double m_lr;
	unsigned int m_num_epochs;
	std::vector<double> m_loss_history;
};
}

#endif // __OPTIMIZER_HPP__

