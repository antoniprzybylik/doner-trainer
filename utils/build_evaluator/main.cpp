#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

enum LayerType {
	LinLayer,
	SigmaLayer,
};

class LayerSpec {
private:
	inline
	std::string sigma_layer_formula_frag(size_t layer_num) const;
	inline
	std::string lin_layer_formula_frag(size_t layer_num,
		std::vector<long double> &params, size_t begin_idx) const;

public:
	enum LayerType layer_type;
	size_t neurons_in, neurons_out;

	LayerSpec(const YAML::Node &node);

	size_t params_cnt(void) const;
	std::string formula_frag(size_t layer_num,
		std::vector<long double> &params, size_t begin_idx) const;
};

static inline
std::pair<const std::string, const std::string>
extract_iospec(const std::string &iospec_str)
{
	size_t delim_pos = 1;
	while (iospec_str[delim_pos] >= '0' &&
	       iospec_str[delim_pos] <= '9') {
		delim_pos++;
	}

	if (delim_pos <= 1 ||
	    iospec_str[delim_pos] != ',' ||
	    iospec_str[delim_pos+1] != ' ') {
		throw std::invalid_argument(
			"Bad layer specification.");
	}

	const std::string neurons_in_str =
		iospec_str.substr(1, delim_pos-1);
	const std::string neurons_out_str =
		iospec_str.substr(delim_pos+2, iospec_str.size()-3-delim_pos);

	return std::make_pair(neurons_in_str,
			      neurons_out_str);
}
	
LayerSpec::LayerSpec(const YAML::Node &node)
{
	const std::string layerspec_str =
		node.as<std::string>();

	if (layerspec_str.size() < 18) {
		throw std::invalid_argument(
			"Bad layer specification.");
	}

	size_t end_of_ident = 0;
	if (layerspec_str.substr(0, 8) ==
	    std::string("LinLayer")) {
		this->layer_type = LinLayer;
		end_of_ident = 8;
	} else if (layerspec_str.substr(0, 10) ==
		   std::string("SigmaLayer")) {
		this->layer_type = SigmaLayer;
		end_of_ident = 10;
	} else {
		throw std::invalid_argument(
			"Bad layer specification.");
	}

	const std::string iospec_str =
		layerspec_str.substr(end_of_ident+4,
				     layerspec_str.size()-1);
	if (*iospec_str.cbegin() != '<' ||
	    *iospec_str.crbegin() != '>') {
		throw std::invalid_argument(
			"Bad layer specification.");
	}

	std::string neurons_in_str;
	std::string neurons_out_str;
	switch (this->layer_type) {
	case LinLayer:
		{
			std::pair<const std::string,
				  const std::string> strs =
				extract_iospec(iospec_str);
	
			neurons_in_str = strs.first;
			neurons_out_str = strs.second;
		}
		break;

	case SigmaLayer:
		neurons_out_str = neurons_in_str =
			iospec_str.substr(1, iospec_str.size()-2);
		break;

	default:
		throw std::runtime_error(
			"Unreachable statement.");
	}

	this->neurons_in = std::stoi(neurons_in_str);
	this->neurons_out = std::stoi(neurons_out_str);
}

size_t LayerSpec::params_cnt(void) const
{
	switch (this->layer_type) {
	case LinLayer:
		return this->neurons_in*
		       this->neurons_out +
		       this->neurons_out;

	case SigmaLayer:
		return 0;

	default:
		throw std::runtime_error(
			"Unreachable statement.");
	}
}

static inline
std::string format_s(size_t layer_num, size_t neuron_num)
{
	return std::string("s_") + std::to_string(layer_num) +
	       std::string("_") + std::to_string(neuron_num);
}

inline
std::string LayerSpec::sigma_layer_formula_frag(size_t layer_num) const
{
	std::string frag;

	for (size_t i = 0; i < this->neurons_out; i++) {
		frag += format_s(layer_num, i) +
			std::string(" = sigma(") +
		        format_s(layer_num-1, i) +
			std::string(");\n");
	}

	return frag;
}

inline
std::string LayerSpec::lin_layer_formula_frag(size_t layer_num,
	std::vector<long double> &params, size_t begin_idx) const
{
	std::string frag;

	for (size_t i = 0; i < this->neurons_out; i++) {
		frag += format_s(layer_num, i) +
			std::string(" = ");

		for (size_t j = 0; j < this->neurons_in; j++) {
			frag += std::string("(") +
				std::to_string(params[begin_idx +
					this->neurons_in*i + j]) +
				std::string(")*") +
				format_s(layer_num-1, j);

			if (j != this->neurons_in-1) {
				frag += std::string(" + ");
			}
		}

		frag += std::string(" + ") +
			std::to_string(params[begin_idx+
				this->neurons_in*this->neurons_out + i]) +
			std::string(";\n");
	}

	return frag;
}

std::string LayerSpec::formula_frag(size_t layer_num,
	std::vector<long double> &params, size_t begin_idx) const
{
	switch (this->layer_type) {
	case LinLayer:
		return lin_layer_formula_frag(layer_num,
			params, begin_idx);

	case SigmaLayer:
		return sigma_layer_formula_frag(layer_num);

	default:
		throw std::runtime_error(
			"Unreachable statement.");
	}
}

int main(const int argc, const char *const argv[])
{
	if (argc < 2) {
		std::cerr << "Error: You did not specify filename."
			  << std::endl;
		return(0);
	}

	if (argc > 2) {
		std::cerr << "Error: Too many arguments."
			  << std::endl;
		return(0);
	}

	const YAML::Node trained_net =
		YAML::LoadFile(argv[1]);

	std::vector<LayerSpec> layers;
	if (trained_net["layers"]) {
		const YAML::Node layer_nodes = trained_net["layers"];
		for (const YAML::Node &layer_node : layer_nodes)
			layers.push_back(LayerSpec(layer_node));
	} else {
		std::cerr << "Error: Malformed data." << std::endl;
		return(0);
	}

	std::vector<long double> params;
	if (trained_net["params"]) {
		const YAML::Node param_nodes = trained_net["params"];
		for (const YAML::Node &param_node : param_nodes)
			params.push_back(param_node.as<long double>());
	} else {
		std::cerr << "Error: Malformed data." << std::endl;
		return(0);
	}

	size_t params_cnt = 0;
	for (LayerSpec &layer : layers)
		params_cnt += layer.params_cnt();

	if (params.size() !=
	    params_cnt) {
		std::cerr << "Error: Wrong number of parameters "
			     "for given network." << std::endl;
		return(0);
	}

	size_t begin_idx = 0;
	for (size_t i = 0; i < layers.size(); i++) {
		std::cout << layers[i].formula_frag(i+1, params, begin_idx) << std::endl;
		begin_idx += layers[i].params_cnt();
	}

	return(0);
}
