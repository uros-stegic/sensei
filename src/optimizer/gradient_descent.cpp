#include <optimizer/gradient_descent.hpp>
#include <model/rnn.hpp>
#include <utils/armadillo_utils.hpp>
#include <cstdlib>

#define forever while(1)

using namespace sensei;

gradient_descent::gradient_descent(loss* l, double lr, unsigned int num_epochs)
	: optimizer(l, lr, num_epochs)
{}

void gradient_descent::optimize(rnn* net, const arma::mat& input, const arma::mat& target)
{
	unsigned int epoch_num = 0;

	auto preds = net->predict(input);
	auto current_loss = m_loss->evaluate(preds, target);
	//std::exit(0);
	m_loss_history.push_back(current_loss);

	forever {
		std::cout << "[Epoch: " << epoch_num+1 << "] loss: " << current_loss << std::endl;

		auto loss_grad = m_loss->gradient(preds, target);
		auto grads = net->back_propagate(loss_grad);
		auto b_grads = grads.first;
		auto w_grads = grads.second;

		std::transform(
			b_grads.begin(),
			b_grads.end(),
			b_grads.begin(),
			[this](arma::mat b) {
				return m_lr*b;
			}
		);
		std::transform(
			w_grads.begin(),
			w_grads.end(),
			w_grads.begin(),
			[this](arma::mat w) {
				return m_lr*w;
			}
		);

		net->update_biases(b_grads);
		net->update_weights(w_grads);

		preds = net->predict(input);
		current_loss = m_loss->evaluate(preds, target);
		m_loss_history.push_back(current_loss);

		if( ++epoch_num >= m_num_epochs ) {
			std::cout << "finished all epochs" << std::endl;
			break;
		}
	}
}
void gradient_descent::summary() const
{
	std::cout << "optimizer: gradient descent" << std::endl;
	m_loss->summary();
}
