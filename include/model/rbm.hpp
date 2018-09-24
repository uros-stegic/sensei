#ifndef __RBM_HPP__
#define __RBM_HPP__

#include <initializer_list>
#include <vector>
#include <memory>
#include <utility>
#include <armadillo>

namespace sensei {
struct optimizer;

class rbm {
public:
	rbm(unsigned int, unsigned int);

	void fit(const arma::mat&, double, unsigned int);
	arma::mat predict(const arma::mat&) const;
	std::vector<arma::mat> contrastive_divergence(const arma::mat&) const;
	void summary() const;
	void update_weights(const std::vector<arma::mat>&);

private:
	unsigned int m_visible_size;
	unsigned int m_hidden_size;
	arma::mat m_weights;
};
}

#endif // __RBM_HPP__

