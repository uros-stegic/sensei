#include <data/data_io.hpp>

arma::mat load_csv(const std::string& filename)
{
	arma::mat data;
	data.load(filename.c_str(), arma::csv_ascii);
	return data;
}

