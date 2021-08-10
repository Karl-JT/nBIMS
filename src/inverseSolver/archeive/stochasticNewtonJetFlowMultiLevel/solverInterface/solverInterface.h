#pragma once

// #include <ChannelCppSolverBiMesh.h>
#include <jetSolver.h>

template<typename problemType>
class forwardProblem{
public:
	problemType turbProp;

	~forwardProblem(){};
	void init(int level, int validation = 0);
};

template<typename problemType>
void forwardProblem<problemType>::init(int level, int validation){
	caseInitialize(turbProp, 550.0, level);
	if (validation == 1){
		turbProp.validationBoundary();
		turbProp.validationPressure();
	}
	initialization(turbProp); 
}

template<typename MCMCType>
class MCMCMethod{
private:
	MCMCType sampler;

public:
	MCMCMethod(string outputPath);
	~MCMCMethod(){};

	template<typename problemType>
	void run(forwardProblem<problemType>& forwardSolver, double QoI[]);

	template<typename problemType>	
	void runValidation(forwardProblem<problemType>& forwardSolver, double QoI[], int numCoef_);
};

template<typename MCMCType>
MCMCMethod<MCMCType>::MCMCMethod(string outputPath):sampler(outputPath){}

template<typename MCMCType>
template<typename problemType>
void MCMCMethod<MCMCType>::run(forwardProblem<problemType>& forwardSolver, double QoI[]){
	sampler.run(forwardSolver, QoI);
}

template<typename MCMCType>
template<typename problemType>
void MCMCMethod<MCMCType>::runValidation(forwardProblem<problemType>& forwardSolver, double QoI[], int numCoef_){
	sampler.numCoef = numCoef_;
	sampler.runValidation(forwardSolver, QoI);
}

