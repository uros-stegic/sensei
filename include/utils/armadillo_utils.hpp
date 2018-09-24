#ifndef __ARMADILLO_UTILS_HPP__
#define __ARMADILLO_UTILS_HPP__

#include <iostream>
#include <utility>
#include <armadillo>
#include <functional>

std::pair<int, int> shape(const arma::mat& m);
std::ostream& operator<<(std::ostream& out, const std::pair<int, int>& p);
arma::mat where(const arma::mat&, const std::function<bool(double)>&, const std::function<double(bool, double)>&);

#endif // __ARMADILLO_UTILS_HPP__

