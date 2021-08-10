#include "turbSolver.h"

reducedSolver::reducedSolver(int level){
	caseInitialize(level, 550.0);
	initialization();
}

void reducedSolver::soluFwd(){
	iterativeSolver();
}

double reducedSolver::lnLikelihood(double sample[], int numCoef){
	double lnLikelihoodt1 = 0;

	for (int i = 0; i < m; i++){
		lnLikelihoodt1 = lnLikelihoodt1 - 1e4*pow(xVelocity[(int)(i*pow(2, l-5))].value()-validationVelocity[(int)(i*pow(2, 2))], 2);
	}
	for (int i = 0; i < m; i++){
		lnLikelihoodt1 = lnLikelihoodt1 - pow(sample[i], 2);
	}
	return lnLikelihoodt1;
}

void reducedSolver::updateParameter(double sample[], int numCoef){
	for (int i = 0; i < m; i++){
		betaML[i] = sample[i];
	}
}


double reducedSolver::adSolver(double sampleProposal[], double gradientVector[], double hessianMatrix[], int numCoef){
	double* xp = new double[m];
	double Py = 0.0;
	for (int i = 0; i < m; i++){
		xp[i] = sampleProposal[i];
	}

	adouble J = 0;
	adouble polyCoef[numCoef];
	trace_on(1);

	for (int i = 0; i < numCoef; i++){
		polyCoef[i] <<= xp[i];
	}
	std::cout << std::endl << "polyCoef: ";
	for (int i = 0; i < numCoef; i++){
		std::cout << polyCoef[i].value() << " ";
	}
	std::cout << std::endl;

	// updateParameter(sampleProposal, numCoef);
	for (int i = 0; i < m; i++){
		betaML[i] = polyCoef[i];
	}

	iterativeSolver();

	for (int i = 0; i < m; i++){
		J = J + 1e4*pow(xVelocity[(int)(i*pow(2, l-5))]-validationVelocity[(int)(i*pow(2, 2))], 2);
	}
	for(int i = 0; i < m; i++){
		J = J + pow(polyCoef[i], 2);
	}
	J >>= Py;
	trace_off(1);
	double* g = new double[numCoef];
	double** h = (double**)malloc(numCoef*sizeof(double*));
	for (int i = 0; i < numCoef; i++){
		h[i] = (double*)malloc((i+1)*sizeof(double));
	}
	gradient(1, numCoef, xp, g);
	hessian(1, numCoef, xp, h);


	for (int i = 0; i < numCoef; i++){
		gradientVector[i] = g[i];
		for (int j = 0; j < i+1; j++){
			hessianMatrix[i*numCoef+j] = h[i][j];			
			hessianMatrix[j*numCoef+i] = h[i][j];
		}
	}

	return Py;
}
