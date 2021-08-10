#pragma once

#include "MCMCBase.h"
#include "pCN.h"

template <typename samplerType, typename solverType> 
class MCMCChain : public MCMCBase {
public:
	int chainIdx = 0;
	int numBurnin = 10;
	int maxChainLength;	
	int sampleSize;
	double alpha;
	double alphaUni;
	double* QoI;
	double* QoIsum;
	std::uniform_real_distribution<double> uniform_distribution{0.0, 1.0};

	//Sampler
	solverType*  solver;
	samplerType* sampler;

	MCMCChain(int maxChainLength_, int sampleSize_, solverType* solver_, double beta_=1.0) : maxChainLength(maxChainLength_+numBurnin), sampleSize(sampleSize_), solver(solver_){
        sampler = new samplerType(solver_, sampleSize_, beta_);
		QoI = new double[sampleSize_];
		QoIsum = new double[sampleSize_];
	};
	virtual ~MCMCChain(){
		delete sampler;
		delete[] QoI;
		delete[] QoIsum;
	};

	virtual void chainInit(std::default_random_engine* generator);
	virtual void sampleProposal();
	virtual void updatelnLikelihood();
	virtual void updateQoI();
	virtual void checkAcceptance(std::default_random_engine* generator);
	virtual void runStep(std::default_random_engine* generator);
	virtual void run(double QoImean[], std::default_random_engine* generator);
};

template <typename samplerType, typename solverType> 
void MCMCChain<samplerType, solverType>::chainInit(std::default_random_engine* generator){
    chainIdx=0;
	double initialSample[sampleSize];
	solver->updateGeneratorSeed(1000*uniform_distribution(*generator));
	solver->priorSample(initialSample);
	for (int i = 0; i < sampleSize; ++i){
		solver->samples[i] = initialSample[i];
	}
	solver->solve();
	solver->solve4Obs();
	lnLikelihoodt0 = solver->lnLikelihood();
	lnLikelihoodt1 = lnLikelihoodt0;
	updateQoI();
}

template <typename samplerType, typename solverType> 
void MCMCChain<samplerType, solverType>::sampleProposal(){
	sampler->sampleProposal();
}

template <typename samplerType, typename solverType> 
void MCMCChain<samplerType, solverType>::updatelnLikelihood(){
	solver->solve4Obs();
	lnLikelihoodt1 = solver->lnLikelihood();
    // std::cout << " lnLikelihoodt1 " << lnLikelihoodt1 << std::endl;
}

template <typename samplerType, typename solverType> 
void MCMCChain<samplerType, solverType>::updateQoI(){
	QoI[0] = solver->solve4QoI();
}

template <typename samplerType, typename solverType> 
void MCMCChain<samplerType, solverType>::checkAcceptance(std::default_random_engine* generator){
	alpha = sampler->getAlpha(lnLikelihoodt0, lnLikelihoodt1);
	alphaUni = log(uniform_distribution(*generator));
	if (alphaUni < alpha){
		std::cout << "sample accpeted" << std::endl;
		accepted = 1;
		acceptedNum += 1;
	} else {
		std::cout << "sample rejected" << std::endl;
		accepted = 0;
        sampler->restoreSample();
	}
}

template <typename samplerType, typename solverType> 
void MCMCChain<samplerType, solverType>::runStep(std::default_random_engine* generator){
	chainIdx += 1;
	sampleProposal();
	solver->solve();
	updatelnLikelihood();
	checkAcceptance(generator);
	if (accepted == 1){
		lnLikelihoodt0 = lnLikelihoodt1;
		updateQoI();
	}
	if (chainIdx > numBurnin){
		QoIsum[0] += QoI[0];
        // std::cout << "MCMCChain " << QoI[0] << std::endl;
	}
	acceptanceRate = acceptedNum/chainIdx;
}

template <typename samplerType, typename solverType> 
void MCMCChain<samplerType, solverType>::run(double QoImean[], std::default_random_engine* generator){
	chainInit(generator);
	QoIsum[0] = 0.0;
	for (int i = 1; i < maxChainLength+1; ++i){
		runStep(generator);
	}
	for (int i = 0; i < sampleSize; ++i){
		QoImean[i] = QoIsum[i]/(maxChainLength-numBurnin);
	}
	std::cout << "acceptanceRate: " << acceptanceRate << std::endl;
}
