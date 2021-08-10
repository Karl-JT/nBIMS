#pragma once

#include <memory>
#include <jetFlow.h>
#include <turbSolver.h>

template <typename T>
class MCMCBase{
private:
public:
	double alpha = 0;
	double lnLikelihoodt0 = 0;
	double lnLikelihoodt1 = 0;
	
	bool accepted;

	std::shared_ptr<T> samplerSolver;

	MCMCBase();
	~MCMCBase();

	virtual void updateParameter();
	virtual void updatelnLikelihood();

	virtual void runStep();
};

template <typename T>
MCMCBase<T>::MCMCBase(){};

template <typename T>
MCMCBase<T>::~MCMCBase(){};

template <typename T>
void MCMCBase<T>::updateParameter(){};

template <typename T>
void MCMCBase<T>::updatelnLikelihood(){};

template <typename T>
void MCMCBase<T>::runStep(){};
