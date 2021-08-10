#include "turbSolver.h"
#include <dataIO.h>

forwardSolver::forwardSolver(int level){
	caseInitialize(level, 550.0);
	initialization();
	
	string refObs = "./refObs.txt";
	updateReference(refObs);
}

void forwardSolver::updateReference(string filePath){
	txt2read(validationYCoordinate, validationVelocity, 289, filePath);
};


void forwardSolver::soluFwd(){
	iterativeSolver();
}

double forwardSolver::lnLikelihood(double sample[], int numCoef){
	int idx = 0;
	double lnLikelihoodt1 = 0;
	double u = 0;

	for (int i = 0; i < 289; i++){
		while (validationYCoordinate[i] > yCoordinate[idx+1]){
			idx++;
		}
		u = xVelocity[idx+1].value() - (xVelocity[idx+1].value() - xVelocity[idx].value())/(yCoordinate[idx+1] - yCoordinate[idx])*(yCoordinate[idx+1] - validationYCoordinate[i]);
		lnLikelihoodt1 = lnLikelihoodt1 - 1e6*pow(u-validationVelocity[i], 2);
	}

	for (int i = 0; i < numCoef; i++){
		lnLikelihoodt1 = lnLikelihoodt1 - pow(sample[i], 2);
	}

	return lnLikelihoodt1;
}

void forwardSolver::updateParameter(double sample[], int numCoef){
	for (int i = 0; i < m; i++){
		betaML[i] = 0; 
		for (int k = 0; k < numCoef; k++){
			betaML[i] = betaML[i] + sample[k]/pow((k+1), 1)*cos(M_PI*(k+1)*yCoordinate[i]); 
		}
		betaML[i] = exp(betaML[i]);
	}
}


double forwardSolver::adSolver(double sampleProposal[], double gradientVector[], double hessianMatrix[], int numCoef){
	double* xp = new double[numCoef];
	double Py = 0.0;
	for (int i = 0; i < numCoef; i++){
		xp[i] = sampleProposal[i];
	}

	adouble J = 0;
	adouble polyCoef[numCoef];

	trace_on(procid);

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
		betaML[i] = 0; 
		for (int k = 0; k < numCoef; k++){
			betaML[i] = betaML[i] + polyCoef[k]/pow((k+1), 1)*cos(M_PI*(k+1)*yCoordinate[i]); 
		}
		betaML[i] = exp(betaML[i]);
	}

	iterativeSolver();

	int idx = 0;
	for (int i = 0; i < 289; i++){
		while (validationYCoordinate[i] > yCoordinate[idx+1]){
			idx++;
		}
		J = J + 1e6*pow((xVelocity[idx+1] - (xVelocity[idx+1] - xVelocity[idx])/(yCoordinate[idx+1] - yCoordinate[idx])*(yCoordinate[idx+1] - validationYCoordinate[i]))-validationVelocity[i], 2);
	}

	for(int i = 0; i < numCoef; i++){
		J = J + pow(polyCoef[i], 2);
	}

	J >>= Py;

	trace_off(procid);

	double* g = new double[numCoef];
	double** h = (double**)malloc(numCoef*sizeof(double*));
	for (int i = 0; i < numCoef; i++){
		h[i] = (double*)malloc((i+1)*sizeof(double));
	}
	gradient(procid, numCoef, xp, g);
	hessian(procid, numCoef, xp, h);

	for (int i = 0; i < numCoef; i++){
		gradientVector[i] = g[i];
		for (int j = 0; j < i+1; j++){
			hessianMatrix[i*numCoef+j] = h[i][j];			
			hessianMatrix[j*numCoef+i] = h[i][j];
		}
	}

	return Py;
}
