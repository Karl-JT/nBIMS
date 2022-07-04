#pragma once

#include "samplerBase.h"

template <typename T>
class pCN : public samplerBase{
public:
	T* forwardSolver;

	int sampleSize;
	double stepSize;
        double* buffer;

	pCN(T* forwardSolver_, int sampleSize_ = 1, double stepSize_ = 1.0) : forwardSolver(forwardSolver_), sampleSize(sampleSize_), stepSize(stepSize_){
        buffer = new double[sampleSize_];
    };
	virtual ~pCN(){
        delete[] buffer;
    };

	void sampleProposal();
        void restoreSample();
	double getAlpha(double lnLikelihoodt0, double lnLikelihoodt1);
};

template <typename T>
void pCN<T>::sampleProposal(){
	double proposal[sampleSize];
	forwardSolver->priorSample(proposal);
	// std::cout << "pCN samples: " << proposal[0] << " step size " << stepSize << " old sample " << forwardSolver->samples[0]  << std::endl;
	for (int i = 0; i < sampleSize; ++i){
        buffer[i] = forwardSolver->samples[i];
		forwardSolver->samples[i] = forwardSolver->samples[i]*sqrt(1.0-pow(stepSize, 2.0))+proposal[i]*stepSize;
	}
	// std::cout << "new samples: "  << forwardSolver->samples[0]  << std::endl;
}

template <typename T>
void pCN<T>::restoreSample(){
	for (int i = 0; i < sampleSize; ++i){
            forwardSolver->samples[i] = buffer[i];
	}
}

template <typename T>
double pCN<T>::getAlpha(double lnLikelihoodt0, double lnLikelihoodt1){
	double a = std::min(0.0, lnLikelihoodt1-lnLikelihoodt0);
	return a;
}

