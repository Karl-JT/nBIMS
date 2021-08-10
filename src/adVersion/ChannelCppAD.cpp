#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>

#include <adolc/adolc.h>
#include <Eigen/Core>
#include "LBFGS/include/LBFGS.h"
#include "../forwardSolverArray/ChannelCppSolverArray.h"
#include "../forwardSolverArray/ChannelCppSolverArray.cpp"

using namespace LBFGSpp;
using namespace std;
using Eigen::VectorXd;


void mainSolver(const adouble betaML[], caseProp<adouble>& turbProp, adouble& J){
	// Parameters
	for (int i = 0; i < turbProp.m; i++){
		turbProp.betaML[i] = betaML[i];
	}
	iterativeSolver(turbProp);
	for (int i = 1; i < turbProp.m; i++) {
		J = J + 1e20*pow((turbProp.xVelocity[i] - turbProp.dnsData[i]*turbProp.frictionVelocity), 2) + 400*pow((turbProp.betaML[i]-1), 2);
	}
};

class optLoop{
private:
	int n;

public:
	caseProp<adouble> turbProp;
	optLoop(int n_) : n(n_){
		caseInitialize(turbProp, 550);
		initialization(turbProp);
	};
	~optLoop(){};

	double operator()(const VectorXd& x, VectorXd& grad){
		double* xp = new double[n];
		double Py = 0.0;
		cout << "beta value" << endl;
		for (int i = 0; i < n; i++){
			xp[i] = x[i];
			cout << x[i] << " ";
		}

		adouble* betaML = new adouble[n];
		adouble J = 0;

		trace_on(1);
		for (int i = 0; i < n; i++){
			betaML[i] <<= xp[i];
		}
		mainSolver(betaML, turbProp, J);
		J >>= Py;
		trace_off();
		double* g = new double[n];
		gradient(1, n, xp, g);

		// ofstream costFunctionFile;
		// costFunctionFile.open("./output/costFunction.csv", ios_base::app);
		// costFunctionFile << J.value() << endl;
		// costFunctionFile.close();

		cout << endl << "gradient" << endl;
		for (int i = 0; i < n; i++){
			cout << g[i] << " ";
			grad[i] = g[i];
		}
		
		cout << endl << "Cost: " << Py << endl; 
		return Py;
	}
};


int main(){
	//Mesh Number
	const int n = 199;

	LBFGSParam<double> param;
	param.epsilon = 1e-6;
	param.max_iterations = 150;

	LBFGSSolver<double> solver(param);
	optLoop optStep(n);

	VectorXd x =  VectorXd::Zero(n);
	for (int i = 0; i < n; i++){
		x[i] = 1.0;
	}
	double fx;
	int niter = solver.minimize(optStep, x, fx);

	cout << niter << " iterations" << endl;
	system("python CppoutputReader.py");

	return 0;
}
