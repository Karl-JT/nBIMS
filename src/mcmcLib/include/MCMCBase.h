#pragma once
#include <memory>

class MCMCBase{
private:
public:
	bool accepted;
	double acceptedNum = 0;
	double acceptanceRate = 0;
	double lnLikelihoodt0 = 0;
	double lnLikelihoodt1 = 0;	

	MCMCBase(){};
	virtual ~MCMCBase(){};

	virtual void chainInit(){};
	virtual void sampleProposal(){};
	virtual void updatelnLikelihood(){};
	virtual void udpateQoI(){};
	virtual void checkAcceptance(){};
	virtual void runStep(){};
};
