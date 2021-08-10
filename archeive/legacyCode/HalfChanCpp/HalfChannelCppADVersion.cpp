#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>

#include <adolc/adolc.h>
#include "LBFGS/include/LBFGS.h"
#include <Eigen/Core>

#include "dnsDataInterpreter.H"

using namespace std;
using Eigen::VectorXd;
using namespace LBFGSpp;

class caseProp {
public:
	int m, simpleGridingRatio, deltaTime;
	double nu, frictionVelocity;
};
caseProp updateCaseProperties(int reTau) {
	caseProp turbProp;
	if (reTau == 180) {
		turbProp.m = 100;
		turbProp.simpleGridingRatio = 100;
		turbProp.deltaTime = 1;
		turbProp.nu = 3.4e-4;
		turbProp.frictionVelocity = 6.37309e-2;
	}
	if (reTau == 550) {
		turbProp.m = 100;
		turbProp.simpleGridingRatio = 150;
		turbProp.deltaTime = 100;
		turbProp.nu = 1e-4;
		turbProp.frictionVelocity = 5.43496e-2;
	}
	if (reTau == 1000) {
		turbProp.m = 100;
		turbProp.simpleGridingRatio = 1000;
		turbProp.deltaTime = 1;
		turbProp.nu = 5e-5;
		turbProp.frictionVelocity = 5.00256e-2;
	}
	if (reTau == 2000) {
		turbProp.m = 100;
		turbProp.simpleGridingRatio = 2500;
		turbProp.deltaTime = 1;
		turbProp.nu = 2.3e-5;
		turbProp.frictionVelocity = 4.58794e-2;
	}
	if (reTau == 5200) {
		turbProp.m = 200;
		turbProp.simpleGridingRatio = 2000;
		turbProp.deltaTime = 1;
		turbProp.nu = 8e-6;
		turbProp.frictionVelocity =4.14872e-2;
	}
	return turbProp;
};
void flowSolver(caseProp& turbProp, vector<double>& yCoordinate, vector<adouble>& xVelocity, vector<adouble>& nut) {
	vector<adouble> vectorA(turbProp.m, 0);
	vector<adouble> vectorB(turbProp.m, 0);
	vector<adouble> vectorC(turbProp.m, 0);
	vector<adouble> vectorD(turbProp.m, 1);
	

	vectorB[0] = -1;
	vectorC[0] = 0;
	vectorA[turbProp.m - 1] = 1;
	vectorB[turbProp.m - 1] = -1;
	vectorD[0] = 0;
	vectorD[turbProp.m - 1] = 0;

	for (int i = 1; i < turbProp.m - 1; i++) {
		vectorA[i] = (2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(turbProp.nu + nut[i - 1] / 2.0 + nut[i] / 2.0) / (yCoordinate[i] - yCoordinate[i - 1]))*turbProp.deltaTime;
		vectorB[i] = (-2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*((turbProp.nu + nut[i] / 2.0 + nut[i + 1] / 2.0) / (yCoordinate[i + 1] - yCoordinate[i]) + (turbProp.nu + nut[i - 1] / 2.0 + nut[i] / 2.0) / (yCoordinate[i] - yCoordinate[i - 1])))*turbProp.deltaTime - 1;
		vectorC[i] = (2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(turbProp.nu + nut[i] / 2.0 + nut[i + 1] / 2.0) / (yCoordinate[i + 1] - yCoordinate[i]))*turbProp.deltaTime;
		vectorD[i] = vectorD[i] * (-pow(turbProp.frictionVelocity, 2)) * turbProp.deltaTime - xVelocity[i];
	}

	vector<adouble> newVectorC(turbProp.m, 0);
	vector<adouble> newVectorD(turbProp.m, 0);

	newVectorC[0] = vectorC[0] / vectorB[0];
	newVectorD[0] = vectorD[0] / vectorB[0];

	for (int i = 1; i < turbProp.m - 1; i++) {
		newVectorC[i] = vectorC[i] / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
		newVectorD[i] = (vectorD[i] - vectorA[i] * newVectorD[i - 1]) / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
	}
	newVectorD[turbProp.m - 1] = (vectorD[turbProp.m - 1] - vectorA[turbProp.m - 1] * newVectorD[turbProp.m - 2]) / (vectorB[turbProp.m - 1] - vectorA[turbProp.m - 1] * newVectorC[turbProp.m - 2]);

	xVelocity[turbProp.m - 1] = newVectorD[turbProp.m - 1];
	for (int i = 0; i < turbProp.m - 1; i++) {
		xVelocity[turbProp.m - 2 - i] = newVectorD[turbProp.m - 2 - i] - newVectorC[turbProp.m - 2 - i] * xVelocity[turbProp.m - 1 - i];
	}
};
void omegaSolver(caseProp& turbProp, vector<double>& yCoordinate, vector<adouble>& xVelocity, vector<adouble>& nut, vector<adouble>& omega, const vector<adouble>& betaML) {
	double sigma = 0.5;
	double alpha = 3.0 / 40.0;
	double gamma = 5.0 / 9.0;
	int boundaryPoints = 6;

	vector<adouble> vectorA(turbProp.m, 0);
	vector<adouble> vectorB(turbProp.m, 0);
	vector<adouble> vectorC(turbProp.m, 0);
	vector<adouble> vectorD(turbProp.m, 0);

	vectorB[0] = -1;
	vectorC[0] = 0;
	vectorD[0] = -1e50;

	for (int i = 1; i < boundaryPoints; i++) {
		vectorA[i] = 0;
		vectorB[i] = -1;
		vectorC[i] = 0;
		vectorD[i] = -6.0 * turbProp.nu / (0.00708 * pow(yCoordinate[i], 2));
	}

	vectorA[turbProp.m - 1] = 1;
	vectorB[turbProp.m - 1] = -1;
	vectorD[turbProp.m - 1] = 0;

	for (int i = boundaryPoints; i < turbProp.m - 1; i++) {
		vectorA[i] = 2.0 * turbProp.deltaTime / (yCoordinate[i + 1] - yCoordinate[i - 1])*(turbProp.nu + sigma * (nut[i] / 2.0 + nut[i - 1] / 2.0)) / (yCoordinate[i] - yCoordinate[i - 1]);
		vectorB[i] = turbProp.deltaTime * (-alpha * omega[i] - 2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*((turbProp.nu + sigma * (nut[i] / 2.0 + nut[i - 1] / 2.0)) / (yCoordinate[i] - yCoordinate[i - 1]) + (turbProp.nu + sigma * (nut[i] / 2.0 + nut[i + 1] / 2.0)) / (yCoordinate[i + 1] - yCoordinate[i]))) - 1;
		vectorC[i] = 2.0 * turbProp.deltaTime / (yCoordinate[i + 1] - yCoordinate[i - 1])*(turbProp.nu + sigma * (nut[i] / 2.0 + nut[i + 1] / 2.0)) / (yCoordinate[i + 1] - yCoordinate[i]);
		vectorD[i] = (-gamma * betaML[i] * pow(((xVelocity[i + 1] - xVelocity[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1])), 2.0))*turbProp.deltaTime - omega[i]; // - max(0, sigmaD / omega[i] * ((k[i + 1] - k[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]))*((omega[i + 1] - omega[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]))))*deltaTime - omega[i];
	}

	vector<adouble> newVectorC(turbProp.m, 0);
	vector<adouble> newVectorD(turbProp.m, 0);

	newVectorC[0] = vectorC[0] / vectorB[0];
	newVectorD[0] = vectorD[0] / vectorB[0];
	for (int i = 1; i < turbProp.m - 1; i++) {
		newVectorC[i] = vectorC[i] / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
		newVectorD[i] = (vectorD[i] - vectorA[i] * newVectorD[i - 1]) / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
	}
	newVectorD[turbProp.m - 1] = (vectorD[turbProp.m - 1] - vectorA[turbProp.m - 1] * newVectorD[turbProp.m - 2]) / (vectorB[turbProp.m - 1] - vectorA[turbProp.m - 1] * newVectorC[turbProp.m - 2]);

	omega[turbProp.m - 1] = newVectorD[turbProp.m - 1];
	for (int i = 0; i < turbProp.m - 1; i++) {
		omega[turbProp.m - 2 - i] = newVectorD[turbProp.m - 2 - i] - newVectorC[turbProp.m - 2 - i] * omega[turbProp.m - 1 - i];
	}
};
void kSolver(caseProp& turbProp, vector<double>& yCoordinate, vector<adouble>& xVelocity, vector<adouble>& nut, vector<adouble>& omega, vector<adouble>& k) {
	double sigmaStar = 0.5;
	double alphaStar = 0.09;

	vector<adouble> vectorA(turbProp.m, 0);
	vector<adouble> vectorB(turbProp.m, 0);
	vector<adouble> vectorC(turbProp.m, 0);
	vector<adouble> vectorD(turbProp.m, 0);

	vectorB[0] = -1.0;
	vectorC[0] = 0.0;
	vectorA[turbProp.m - 1] = 1.0;
	vectorB[turbProp.m - 1] = -1.0;
	vectorD[0] = 0;
	vectorD[turbProp.m - 1] = 0;

	for (int i = 1; i < turbProp.m - 1; i++){
		vectorA[i] = 2.0 * turbProp.deltaTime / (yCoordinate[i + 1] - yCoordinate[i - 1])*(turbProp.nu + sigmaStar * (nut[i] / 2 + nut[i - 1] / 2)) / (yCoordinate[i] - yCoordinate[i - 1]);
		vectorB[i] = turbProp.deltaTime * (-alphaStar * omega[i] - 2 / (yCoordinate[i + 1] - yCoordinate[i - 1])*((turbProp.nu + sigmaStar * (nut[i] / 2 + nut[i - 1] / 2)) / (yCoordinate[i] - yCoordinate[i - 1]) + (turbProp.nu + sigmaStar * (nut[i] / 2 + nut[i + 1] / 2)) / (yCoordinate[i + 1] - yCoordinate[i]))) - 1;
		vectorC[i] = 2.0 * turbProp.deltaTime / (yCoordinate[i + 1] - yCoordinate[i - 1])*(turbProp.nu + sigmaStar * (nut[i] / 2 + nut[i + 1] / 2)) / (yCoordinate[i + 1] - yCoordinate[i]);
		vectorD[i] = (-nut[i] * pow((xVelocity[i + 1] - xVelocity[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]), 2.0))*turbProp.deltaTime - k[i];
	}
	vector<adouble> newVectorC(turbProp.m);
	vector<adouble> newVectorD(turbProp.m);

	newVectorC[0] = vectorC[0] / vectorB[0];
	newVectorD[1] = vectorD[0] / vectorB[0];
	for (int i = 1; i < turbProp.m - 1; i++) {
		newVectorC[i] = vectorC[i] / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
		newVectorD[i] = (vectorD[i] - vectorA[i] * newVectorD[i - 1]) / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
	}
	newVectorD[turbProp.m - 1] = (vectorD[turbProp.m - 1] - vectorA[turbProp.m - 1] * newVectorD[turbProp.m - 2]) / (vectorB[turbProp.m - 1] - vectorA[turbProp.m - 1] * newVectorC[turbProp.m - 2]);

	k[turbProp.m - 1] = newVectorD[turbProp.m - 1];
	for (int i = 0; i < turbProp.m - 1; i++) {
		k[turbProp.m - 2 - i] = newVectorD[turbProp.m - 2 - i] - newVectorC[turbProp.m - 2 - i] * k[turbProp.m - 1 - i];
	}
};
void mainSolver(const vector<adouble>& betaML, adouble& J){
	// Parameters
	double reTau = 5200;
	caseProp turbProp = updateCaseProperties(reTau);

	// Node Coordinate
	vector<double> yCoordinate(turbProp.m);

	// Variables on Nodes
	vector<adouble> xVelocity(turbProp.m, 0);
	vector<adouble> k(turbProp.m, 1e-8);
	vector<adouble> omega(turbProp.m, 1e5);
	vector<adouble> nut(turbProp.m, 1e-5);

	// Node Coordinate Initialization
	double gridRatio = pow(turbProp.simpleGridingRatio, 1.0 / (turbProp.m - 2));
	double firstNodeDist = (1 - gridRatio) / (1 - pow(gridRatio, turbProp.m - 1));

	double tempGridSize = firstNodeDist;
	yCoordinate[0] = 0;
	for (int i = 1; i < turbProp.m; i++) {
		yCoordinate[i] = yCoordinate[i - 1] + tempGridSize;
		tempGridSize = tempGridSize * gridRatio;
	}
	dnsDataInterpreter dnsData(yCoordinate, reTau);
	for (int i = 0; i < 100000; i++) {
		//cout << "#";
		vector<adouble> xVelocityTemp = xVelocity;
		flowSolver(turbProp, yCoordinate, xVelocity, nut);
		omegaSolver(turbProp, yCoordinate, xVelocity, nut, omega, betaML);
		kSolver(turbProp, yCoordinate, xVelocity, nut, omega, k);
		adouble res = 0;
		for (int i = 0; i < turbProp.m; i++) {
			nut[i] = k[i] / omega[i];
			res = res + pow(xVelocity[i] - xVelocityTemp[i], 2);
		}
		if (res < 1e-9){
			cout << endl << "res: " << res << endl;
			break;
		}
	}
	ofstream yFile;
	ofstream uFile;
	ofstream betaFile;
	yFile.open("yFile.csv");
	uFile.open("uFile.csv");
	betaFile.open("betaFile.csv");
	for (int i = 0; i < yCoordinate.size(); i++) {
		yFile << yCoordinate[i] << endl;
		uFile << xVelocity[i].value() << endl;
		betaFile << betaML[i].value() << endl;
	}
	for (int i = 1; i < yCoordinate.size()-1; i++) {
		J = J + 1e20*pow((xVelocity[i]-dnsData.U[i]*turbProp.frictionVelocity)*(yCoordinate[i]-yCoordinate[i-1]), 2) + 1*pow((betaML[i]-1)*(yCoordinate[i]-yCoordinate[i-1]), 2);
	}
};

class optLoop{
private:
	int n;
public:
	optLoop(int n_) : n(n_) {}
	double operator()(const VectorXd& x, VectorXd& grad){
		double* xp = new double[n];
		double Py = 0.0;
		cout << "beta value" << endl;
		for (int i = 0; i < n; i++){
			xp[i] = x[i];
			cout << x[i] << " ";
		}

		adouble* betaMl = new adouble[n];
		adouble J = 0;

		trace_on(1);
		vector<adouble> betaML(n);
		for (int i = 0; i < 200; i++){
			betaML[i] <<= xp[i];
		}
		mainSolver(betaML, J);
		J >>= Py;
		trace_off();
		double* g = new double[n];
		gradient(1, n, xp, g);

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
	const int n = 200;

	LBFGSParam<double> param;
	param.epsilon = 1e-6;
	param.max_iterations = 500;

	LBFGSSolver<double> solver(param);
	optLoop optStep(n);

	VectorXd x =  VectorXd::Zero(200);
	for (int i = 0; i < 200; i++){
		x[i] = 1.0;
	}
	double fx;
	int niter = solver.minimize(optStep, x, fx);

	cout << niter << " iterations" << endl;
	system("python CppoutputReader.py");

	return 0;
}