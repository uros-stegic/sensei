#include <optimizer/optimizer.hpp>

using namespace sensei;

optimizer::optimizer(loss* l, double lr, unsigned int num_epochs)
	: m_loss(l)
	, m_lr(lr)
	, m_num_epochs(num_epochs)
{}
std::vector<double> optimizer::loss_history() const
{
	return m_loss_history;
}

