#ifndef __CELL_HPP__
#define __CELL_HPP__

#include <armadillo>
#include <memory>
#include <sstream>
#include <string>

#include <activations/activation.hpp>
#include <activations/tanh.hpp>
#include <activations/sigmoid.hpp>

namespace sensei {
class cell {
public:
	cell(unsigned int, activation*, double=0);

	arma::mat forward_pass(const arma::mat&);
	arma::mat backward_pass(const arma::mat&);

	unsigned int init_weights(int prev_layer);
	void summary() const;
	arma::mat get_b_grad() const;
	arma::mat get_w_grad() const;
	void update_bias(const arma::mat&);
	void update_weights(const arma::mat&);

private:
	unsigned int m_neurons_size;
	std::shared_ptr<activation> m_activation;
	double m_dropout;

	arma::mat m_bias;
	arma::mat m_weights;
	arma::mat m_last_input;
	arma::mat m_hidden_state;
	arma::mat m_b_grad;
	arma::mat m_w_grad;
};

template<typename Activation, int NUnits>
std::shared_ptr<cell> make_cell(double dropout=0)
{
	return std::make_shared<cell>(NUnits, new Activation, dropout);
}
}

#endif // __CELL_HPP__

