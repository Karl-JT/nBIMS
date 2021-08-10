#include "jetFlow.h"

double jetFlow::lnLikelihood(double sample[], int numCoef){
	double lnLikelihood = misfit();
	return lnLikelihood;
};

void jetFlow::adSolver(double sampleProposal[], double gradientVector[], double hessianMatrix[], int numCoef){
	soluFwd();
	soluAdj();
	misfit();
	grad(gradientVector);

	std::shared_ptr<Matrix> hessianFEnicsMatrix;
	hessianFEnicsMatrix = hessian(hessianMatrix);
};
