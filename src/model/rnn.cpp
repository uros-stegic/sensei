#include <model/rnn.hpp>
#include <optimizer/optimizer.hpp>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <utils/armadillo_utils.hpp>

using namespace sensei;

rnn::rnn(unsigned int input_size, const std::initializer_list<std::shared_ptr<cell>>& cells, optimizer* opt)
	: m_input_size(input_size)
	, m_cells(cells)
	, m_optimizer(opt)
{
	unsigned int prev_size = m_input_size;
	for(auto&& m_cell : m_cells) {
		prev_size = m_cell->init_weights(prev_size);
	}
}

void rnn::fit(const arma::mat& input, const arma::mat& target)
{
	m_optimizer->optimize(this, input, target);
}

arma::mat rnn::predict(const arma::mat& input) const
{
	return std::accumulate(
		m_cells.begin(),
		m_cells.end(),
		input,
		[](const arma::mat& acc, const std::shared_ptr<cell>& it) {
			return it->forward_pass(acc);
		}
	);
}

std::pair<std::vector<arma::mat>, std::vector<arma::mat>> rnn::back_propagate(const arma::mat& grad) const
{
	std::accumulate(
		m_cells.rbegin(),
		m_cells.rend(),
		grad,
		[](const arma::mat& acc, const std::shared_ptr<cell>& it) {
			return it->backward_pass(acc);
		}
	);

	std::vector<arma::mat> b_grads;
	std::transform(
		m_cells.begin(),
		m_cells.end(),
		std::back_inserter(b_grads),
		[](const std::shared_ptr<cell>& c) {
			return c->get_b_grad();
		}
	);
	std::vector<arma::mat> w_grads;
	std::transform(
		m_cells.begin(),
		m_cells.end(),
		std::back_inserter(w_grads),
		[](const std::shared_ptr<cell>& c) {
			return c->get_w_grad();
		}
	);
	
	return std::make_pair(b_grads, w_grads);
}

void rnn::update_biases(const std::vector<arma::mat>& updates)
{
	for(unsigned int i = 0; i < updates.size(); i++) {
		m_cells[i]->update_bias(updates[i]);
	}
}
void rnn::update_weights(const std::vector<arma::mat>& updates)
{
	for(unsigned int i = 0; i < updates.size(); i++) {
		m_cells[i]->update_weights(updates[i]);
	}
}

void rnn::summary() const
{
	std::cout << "Recurrent Neural Network" << std::endl;
	std::cout << "Input size: " << m_input_size << std::endl;
	std::cout << "**************" << std::endl;
	for(auto&& m_cell : m_cells) {
		m_cell->summary();
		std::cout << "**************" << std::endl;
	}
	m_optimizer->summary();
}

