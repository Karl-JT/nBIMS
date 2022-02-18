#include "adCore.h"

#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
using namespace std;

double adIterativeSolver(caseProp<adouble>& turbProp, double approximateCoefProposal[], double gradientVector[], double hessianMatrix[], int numCoef){
	double* xp = new double[numCoef];
	double Py = 0.0;
	for (int i = 0; i < numCoef; i++){
		xp[i] = approximateCoefProposal[i];
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

	for (int i = 0; i < turbProp.m; i++){
		turbProp.betaML[i] = 0;
		for (int j = 0; j < numCoef; j++){
			turbProp.betaML[i] = turbProp.betaML[i]+ cos(M_PI*(j+1)*turbProp.yCoordinate[i])*polyCoef[j]/pow((j+1), 1);
		}
		turbProp.betaML[i] = exp(turbProp.betaML[i]);
 	}

	iterativeSolver(turbProp);

	for (int i = 0; i < turbProp.m; i++){
		if (abs(turbProp.yCoordinate[i]+0.75) < 1e-8){
			J = J - 1e4*pow(turbProp.xVelocity[i]-0.9081, 2);
		}else if (abs(turbProp.yCoordinate[i]+0.5) < 1e-8){
			J = J - 1e4*pow(turbProp.xVelocity[i]-0.9554, 2);
		} else if (abs(turbProp.yCoordinate[i]) < 1e-8){
			J = J - 1e4*pow(turbProp.xVelocity[i]-1.0571, 2);			
		} else if (abs(turbProp.yCoordinate[i]-0.5) < 1e-8){
			J = J - 1e4*pow(turbProp.xVelocity[i]-0.9683, 2);			
		} else if (abs(turbProp.yCoordinate[i]-0.75) < 1e-8){
			J = J - 1e4*pow(turbProp.xVelocity[i]-0.8875, 2);			
		}
	}

	for(int i = 0; i < numCoef; i++){
		J = J - pow(polyCoef[i], 2);
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
