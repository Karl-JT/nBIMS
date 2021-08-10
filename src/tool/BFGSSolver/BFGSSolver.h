#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>

#include <adolc/adolc.h>
#include <Eigen/Core>
#include <LBFGS.h>
#include <turbSolver.h>

using namespace LBFGSpp;
using namespace std;
using Eigen::VectorXd;


class optLoop{
public:
	int numCoef;
	forwardSolver* turbProp;
	optLoop(forwardSolver& solverReference, int dimension){
		turbProp = &solverReference;
		numCoef = dimension;
	};
	~optLoop(){};

	double operator()(const VectorXd& x, VectorXd& grad){
		double* xp = new double[numCoef];
		double Py = 0.0;
		cout << "Coef: ";
		for (int i = 0; i < numCoef; i++){
			xp[i] = x[i];
			cout << x[i] << " ";
		}
		std::cout << std::endl;

		adouble* coef = new adouble[numCoef];
		adouble J = 0;

		trace_on(1);

		for (int i = 0; i < numCoef; i++){
			coef[i] <<= xp[i];
		}

		///**************mainSolver*****************************
		for (int i = 0; i < turbProp->m; i++){
			turbProp->betaML[i] = 0;
			for (int j = 0; j < numCoef/2.0; j++){
				turbProp->betaML[i] = turbProp->betaML[i]+ cos(M_PI*coef[2*j+1]*turbProp->yCoordinate[i])*coef[2*j]/pow((j+1), 1);
			}
			turbProp->betaML[i] = exp(turbProp->betaML[i]);
	 	}
	 	turbProp->iterativeSolver();
		for (int i = 0; i < turbProp->m; i++) {
			J = J + turbProp->nodeWeight[i]*1e20*pow((turbProp->xVelocity[i] - turbProp->dnsData[i]*turbProp->frictionVelocity), 2);
		}
		for (int i = 0; i < numCoef/2; i++){
			J = J  + 100*pow(coef[2*i], 2);
		}
		///**************mainSolver****************************


		J >>= Py;
		trace_off();
		double* g = new double[numCoef];
		gradient(1, numCoef, xp, g);

		for (int i = 0; i < numCoef; i++){
			grad[i] = g[i];
		}
		
		cout << endl << "Cost: " << Py << endl; 
		return Py;
	}
};

template<typename problemType>
class BFGSSolver{
private:
public:
	~BFGSSolver(){};

	void findMAP(problemType& forwardSolver, int dimension, double QoI[]);
};

template<typename problemType>
void BFGSSolver<problemType>::findMAP(problemType& forwardSolver, int dimension, double QoI[]){
	optLoop optStep(forwardSolver, dimension);
	//Mesh Number
	LBFGSParam<double> param;
	param.epsilon = 1e-6;
	param.max_iterations = 100;

	LBFGSSolver<double> solver(param);
	//optLoop optStep(forwardSolver.turbProp.m);

	VectorXd x =  VectorXd::Zero(dimension);
	for (int i = 0; i < dimension/2; i++){
		x[2*i]   = 0;
		x[2*i+1] = pow(2, i);
	}
	double fx;
	int niter = solver.minimize(optStep, x, fx);

	cout << niter << " iterations" << endl;
	for (int i = 0; i < dimension; i++){
		QoI[i] = x[i];
	}
}
