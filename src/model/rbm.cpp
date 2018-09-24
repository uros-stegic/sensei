#include <model/rbm.hpp>
#include <optimizer/optimizer.hpp>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <utils/armadillo_utils.hpp>
#include <activations/sigmoid.hpp>
#include <random>

using namespace sensei;

rbm::rbm(unsigned int visible_size, unsigned int hidden_size)
	: m_visible_size(visible_size)
	, m_hidden_size(hidden_size)
{
	auto interval = 0.1 * std::sqrt(6 / (m_visible_size + m_hidden_size));
	m_weights = 2*interval*arma::randu<arma::mat>(m_visible_size+1, m_hidden_size+1) - interval;
}

void rbm::fit(const arma::mat& data, double lr, unsigned int epochs_count)
{
	arma::mat biased_data = arma::join_horiz(arma::ones(data.n_rows), data);
	auto act = sigmoid();
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<> uniform(0, 1);
	for(unsigned int epoch = 0; epoch < epochs_count; epoch++) {
		// positive phase
		arma::mat positive_logits = biased_data*m_weights;
		arma::mat positive_activations = act.evaluate(positive_logits);
		positive_activations.col(0) = arma::ones<arma::mat>(positive_activations.n_rows, 1);
		arma::mat sampled_latent = where(
				positive_activations, 
				[&uniform=uniform, &rng=rng](double d) { return d > uniform(rng); },
				[](bool b, double d) { return b ? 1 : 0; }
		);
		arma::mat positive_association = biased_data.t() * positive_activations;

		// negative phase
		arma::mat negative_logits = sampled_latent*m_weights.t();
		arma::mat negative_activations = act.evaluate(negative_logits);
		negative_activations.col(0) = arma::ones<arma::mat>(negative_activations.n_rows, 1);

		arma::mat negative_latent_logits = negative_activations * m_weights;
		arma::mat negative_latent_activations = act.evaluate(negative_latent_logits);
		arma::mat negative_association = negative_activations.t() * negative_latent_activations;

		m_weights += lr * ((positive_association - negative_association)/data.n_rows);

		std::cout << "epoch: " << epoch << " error: " << arma::accu(arma::pow((biased_data - negative_activations), 2)) << std::endl;
	}
}

arma::mat rbm::predict(const arma::mat& input) const
{
	return input;
}

std::vector<arma::mat> rbm::contrastive_divergence(const arma::mat& grad) const
{
	return {grad};
}

void rbm::summary() const
{
	std::cout << "Restricted Boltzmann Machine" << std::endl;
	std::cout << "Input variable size: " << m_visible_size << std::endl;
	std::cout << "Latent variable size: " << m_hidden_size << std::endl;
	std::cout << "**************" << std::endl;
}

