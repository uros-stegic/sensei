#include <model/cell.hpp>
#include <utils/armadillo_utils.hpp>
#include <cstdlib>

using namespace sensei;

cell::cell(unsigned int n_neurons, activation* a, double dropout)
	: m_neurons_size(n_neurons)
	, m_activation(a)
	, m_dropout(dropout)
{}

arma::mat cell::forward_pass(const arma::mat& input)
{
	m_last_input = input;
	m_hidden_state = m_activation->evaluate(m_weights * input + arma::repmat(m_bias, 1, input.n_cols));
	return m_hidden_state;
}

arma::mat cell::backward_pass(const arma::mat& err)
{
	arma::mat d = err % m_activation->gradient(m_hidden_state);
	m_b_grad = arma::sum(d, 1);
	m_w_grad = d * m_last_input.t();

	return m_weights.t() * d;
}
void cell::update_bias(const arma::mat& b_update)
{
	m_bias = m_bias - b_update;
}
void cell::update_weights(const arma::mat& w_update)
{
	m_weights = m_weights - w_update;
}

arma::mat cell::get_w_grad() const
{
	return m_w_grad;
}
arma::mat cell::get_b_grad() const
{
	return m_b_grad;
}

unsigned int cell::init_weights(int prev_layer)
{
	m_bias = (2*arma::randu<arma::mat>(m_neurons_size, 1)-1)*std::sqrt(m_neurons_size);
	m_weights = (2*arma::randu<arma::mat>(m_neurons_size, prev_layer)-1)*std::sqrt(m_neurons_size);
	return m_neurons_size;
}

void cell::summary() const
{
	std::cout << "cell: " << std::endl;
	std::cout << "bias shape: " << shape(m_bias) << std::endl;
	std::cout << "weights shape: " << shape(m_weights) << std::endl;
	std::cout << "number of units: " << m_neurons_size << std::endl;
	if( m_dropout > 0 ) {
		std::cout << "using dropout: " << m_dropout << std::endl;
	}
	m_activation->summary();
}

