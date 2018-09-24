#ifndef __RNN_HPP__
#define __RNN_HPP__

#include <initializer_list>
#include <vector>
#include <memory>
#include <utility>

#include <model/model.hpp>
#include <model/cell.hpp>

namespace sensei {
struct optimizer;

class rnn : public model {
public:
	rnn(unsigned int, const std::initializer_list<std::shared_ptr<cell>>&, optimizer*);

	void fit(const arma::mat&, const arma::mat&) override;
	arma::mat predict(const arma::mat&) const override;
	std::pair<std::vector<arma::mat>, std::vector<arma::mat>> back_propagate(const arma::mat&) const;
	void summary() const override;
	void update_biases(const std::vector<arma::mat>&);
	void update_weights(const std::vector<arma::mat>&);

private:
	unsigned int m_input_size;
	std::vector<std::shared_ptr<cell>> m_cells;
	std::shared_ptr<optimizer> m_optimizer;
};
}

#endif // __RNN_HPP__

