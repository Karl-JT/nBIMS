#pragma once

#include "MCMCChain.h"

template <typename samplerType, typename solverType> 
class MLChain00 : public MCMCChain<samplerType, solverType> {
public:
	MLChain00(int maxChainLength_, int sampleSize_, solverType* solver_) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_){};
	~MCMCChain(){};
};
