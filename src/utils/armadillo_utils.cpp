#include <utils/armadillo_utils.hpp>

std::pair<int, int> shape(const arma::mat& m)
{
	return std::make_pair(m.n_rows, m.n_cols);
}
std::ostream& operator<<(std::ostream& out, const std::pair<int, int>& p)
{
	return out << "(" << p.first << ", " << p.second << ")";
}

arma::mat where(const arma::mat& M, const std::function<bool(double)>& predicate, const std::function<double(bool, double)>& action)
{
	arma::mat res = M;
	for(unsigned int i = 0; i < M.n_rows; i++) {
		for(unsigned int j = 0; j < M.n_cols; j++) {
			res(i, j) = action(predicate(res(i, j)), res(i, j));
		}
	}
	return res;
}

