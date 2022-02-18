#pragma once

#include <MCMCMetropolis.h>
#include "MCMCBase.h"

template <typename T> 
class MCMCChain : public MCMCBase<T> {
public:
	int chainLength = 0;
	int maxChainLength;
	
	int procid;
	int numCoef;
	int samplerLevel;

	//Sampler
	std::shared_ptr<chainIO> MCMCChainIO;
	std::unique_ptr<double[]> sampleCurrent;
	std::unique_ptr<double[]> sampleProposal;

	MCMCChain(string outputPath, int MLl_, int procid_, int numCoef_, int samplerLevel_);
	~MCMCChain(){};

	virtual void updateParameter(double sample[]);
	virtual void updatelnLikelihood(double sample[]);

	virtual void solve();
	virtual void runStep(){};
	virtual void run(){};

	double returnLikelihoodref(double externalSample[]);
};

template <typename T>
MCMCChain<T>::MCMCChain(string outputPath, int maxChainLength_, int procid_, int numCoef_, int samplerLevel_) : MCMCBase<T>(), maxChainLength(maxChainLength_), procid(procid_), numCoef(numCoef_), samplerLevel(samplerLevel_) {
	MCMCChainIO   = std::make_shared<chainIO>(outputPath, procid_, numCoef_);
	this->samplerSolver = std::make_shared<T>(samplerLevel_);
	this->samplerSolver->procid = procid_;

	sampleCurrent = std::make_unique<double[]>(numCoef);
	sampleProposal = std::make_unique<double[]>(numCoef);
}

template <typename T>
void MCMCChain<T>::solve(){
	this->samplerSolver->soluFwd();
}

template <typename T>
void MCMCChain<T>::updateParameter(double sample[]){
	this->samplerSolver->updateParameter(sample, numCoef);
}

template <typename T>
void MCMCChain<T>::updatelnLikelihood(double sample[]){
	this->samplerSolver->lnLikelihood(sample, numCoef);
}

template <typename T>
double MCMCChain<T>::returnLikelihoodref(double externalSample[]){
	updateParameter(externalSample);
	solve();
	return this->samplerSolver->lnLikelihood(externalSample, numCoef);
}
