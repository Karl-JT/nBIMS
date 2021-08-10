#include "confIO.h"

void read_config_file(std::vector<std::string> &paras, std::vector<std::string> &vals)
{
	std::ifstream cFile("mlmcmc_config.txt");
	if (cFile.is_open()){
		std::string line;
		while(getline(cFile, line)){
			line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
			if (line[0] == '#' || line.empty()){
				continue;
			}
			auto delimiterPos = line.find("=");
			auto name = line.substr(0, delimiterPos);
			auto value = line.substr(delimiterPos+1);

			paras.push_back(name);
			vals.push_back(value);
		}
	}
	cFile.close();
}

void read_config(conf* confData)
{
	read_config_file(confData->name, confData->value);

	confData->num_term = std::stoi(confData->value[0]);
	for (int i = 0; i < confData->num_term; ++i){
		confData->rand_coef.push_back(std::stod(confData->value[i+1]));
	}

	confData->levels                  = std::stoi(confData->value[confData->num_term+1]);
	confData->a                       = std::stoi(confData->value[confData->num_term+2]);
	confData->pCNstep                 = std::stod(confData->value[confData->num_term+3]);
	confData->task                    = std::stoi(confData->value[confData->num_term+4]);
	confData->parallelChain           = std::stoi(confData->value[confData->num_term+5]);
	confData->plainMCMC_sample_number = std::stoi(confData->value[confData->num_term+6]);
	confData->obsNumPerRow            = std::stoi(confData->value[confData->num_term+7]);
	confData->noiseVariance           = std::stod(confData->value[confData->num_term+8]);
	confData->randomSeed              = std::stoi(confData->value[confData->num_term+9]);
}

void display_config(conf* confData)
{
		std::cout << std::endl;
		std::cout << "levels: "           << confData->levels                  <<  std::endl;
		std::cout << "a: "                << confData->a                       <<  std::endl;
		std::cout << "pCNstep: "          << confData->pCNstep                 <<  std::endl;
		std::cout << "task: "             << confData->task                    <<  std::endl;
		std::cout << "parallelChain: "    << confData->parallelChain           <<  std::endl;
		std::cout << "plainMCMC samples:" << confData->plainMCMC_sample_number <<  std::endl;
		std::cout << "obsNumPerRow: "     << confData->obsNumPerRow            <<  std::endl;
		std::cout << "noiseVariance: "    << confData->noiseVariance           <<  std::endl; 
		std::cout << "randomSeed: "       << confData->randomSeed              <<  std::endl;    
};