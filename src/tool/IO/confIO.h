#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

struct conf
{
    std::vector<std::string> name;
	std::vector<std::string> value;
	std::vector<double> rand_coef;

    int num_term,levels,a,task,parallelChain,plainMCMC_sample_number,obsNumPerRow;
    double pCNstep,noiseVariance,randomSeed;
};

void read_config_file(std::vector<std::string> &paras, std::vector<std::string> &vals);

void read_config(conf* confData);

void display_config(conf* confData);

