#include "ChannelCppSolverNoAD.h"
#include <cmath>
#include <iostream>

using namespace std;

void meshGen(int simpleGridingRatio, int m, double reTau, vector<double> & yCoordinate){
        // Node Coordinate Initialization
        double gridRatio = pow(simpleGridingRatio, 1.0/((m+1)/2.0 - 2));
        double firstNodeDist = (1-gridRatio) / (1-pow(gridRatio, (m+1)/2.0 - 1));
        double tempGridSize = firstNodeDist;
        yCoordinate[0] = -1;
        for (int i = 1; i < (m+1)/2.0-1; i++) {
                yCoordinate[i] = yCoordinate[i - 1] + tempGridSize;
                tempGridSize = tempGridSize * gridRatio;
        }
        yCoordinate[(m+1)/2.0] = 0;
        for (int i = (m+1)/2.0; i < m; i++){
                yCoordinate[i] = -yCoordinate[m-1-i];
        }
}

caseProp::caseProp(double reTau){
	//initialize reTau number and update corresponding properties
	if (reTau == 180) {
		m = 199;
		simpleGridingRatio = 80;
		deltaTime = 1;
		nu = 3.4e-4;
		frictionVelocity = 6.37309e-2;
	}
	if (reTau == 550) {
		m = 199;
		simpleGridingRatio = 300;
		deltaTime = 1;
		nu = 1e-4;
		frictionVelocity = 5.43496e-2;
	}
	if (reTau == 1000) {
		m = 199;
		simpleGridingRatio = 1500;
		deltaTime = 1;
		nu = 5e-5;
		frictionVelocity = 5.00256e-2;
	}
	if (reTau == 2000) {
		m = 199;
		simpleGridingRatio = 3000;
		deltaTime = 1;
		nu = 2.3e-5;
		frictionVelocity = 4.58794e-2;
	}
	if (reTau == 5200) {
		m = 199;
		simpleGridingRatio = 10000;
		deltaTime = 1;
		nu = 8e-6;
		frictionVelocity =4.14872e-2;
	}
	//update mesh dimensions according to reTau
	yCoordinate.resize(m);
	xVelocity.resize(m, 0);
	initXVelocity.resize(m, 0);
	dnsData.resize(m);
	k.resize(m, 1e-8);
	omega.resize(m, 1e5);
	nut.resize(m, 1e-5);
	betaML.resize(m, 1);

	//update mesh data on yCoordinate
	meshGen(simpleGridingRatio, m, reTau, yCoordinate);

	//update interpreted DNS data
	dnsDataInterpreter dnsDataTable(yCoordinate, reTau);
    dnsData = dnsDataTable.U;
}

void thomasSolver(vector<double>& vectorA, vector<double>& vectorB, vector<double>& vectorC, vector<double>& vectorD, vector<double>& solution){
	int vectorSize = vectorD.size();

	vector<double> newVectorC(vectorSize, 0);
	vector<double> newVectorD(vectorSize, 0);

	newVectorC[0] = vectorC[0] / vectorB[0];
	newVectorD[0] = vectorD[0] / vectorB[0];

	for (int i = 1; i < vectorSize - 1; i++) {
		newVectorC[i] = vectorC[i] / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
		newVectorD[i] = (vectorD[i] - vectorA[i] * newVectorD[i - 1]) / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
	}
	newVectorD[vectorSize - 1] = (vectorD[vectorSize - 1] - vectorA[vectorSize - 1] * newVectorD[vectorSize - 2]) / (vectorB[vectorSize - 1] - vectorA[vectorSize - 1] * newVectorC[vectorSize - 2]);

	solution[vectorSize - 1] = newVectorD[vectorSize - 1];
	for (int i = 0; i < vectorSize - 1; i++) {
		solution[vectorSize - 2 - i] = newVectorD[vectorSize - 2 - i] - newVectorC[vectorSize - 2 - i] * solution[vectorSize - 1 - i];
	}
}

void flowSolver(caseProp& turbProp) {
	vector<double> vectorA(turbProp.m, 0);
	vector<double> vectorB(turbProp.m, 0);
	vector<double> vectorC(turbProp.m, 0);
	vector<double> vectorD(turbProp.m, 1);
	

	vectorB[0] = -1;
	vectorC[0] = 0;
	vectorA[turbProp.m - 1] = 0;
	vectorB[turbProp.m - 1] = -1;
	vectorD[0] = 0;
	vectorD[turbProp.m - 1] = 0;

	for (int i = 1; i < turbProp.m - 1; i++) {
		vectorA[i] = (2.0 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*(turbProp.nu + turbProp.nut[i - 1] / 2.0 + turbProp.nut[i] / 2.0) / (turbProp.yCoordinate[i] - turbProp.yCoordinate[i - 1]))*turbProp.deltaTime;
		vectorB[i] = (-2.0 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*((turbProp.nu + turbProp.nut[i] / 2.0 + turbProp.nut[i + 1] / 2.0) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i]) + (turbProp.nu + turbProp.nut[i - 1] / 2.0 + turbProp.nut[i] / 2.0) / (turbProp.yCoordinate[i] - turbProp.yCoordinate[i - 1])))*turbProp.deltaTime - 1;
		vectorC[i] = (2.0 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*(turbProp.nu + turbProp.nut[i] / 2.0 + turbProp.nut[i + 1] / 2.0) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i]))*turbProp.deltaTime;
		vectorD[i] = vectorD[i] * (-pow(turbProp.frictionVelocity, 2)) * turbProp.deltaTime - turbProp.xVelocity[i];
	}
	thomasSolver(vectorA, vectorB, vectorC, vectorD, turbProp.xVelocity);
}

void omegaSolver(caseProp& turbProp) {
	double sigma = 0.5;
	double alpha = 3.0 / 40.0;
	double gamma = 5.0 / 9.0;
	int boundaryPoints = 3;

	vector<double> vectorA(turbProp.m, 0);
	vector<double> vectorB(turbProp.m, 0);
	vector<double> vectorC(turbProp.m, 0);
	vector<double> vectorD(turbProp.m, 0);

	vectorB[0] = -1;
	vectorC[0] = 0;
	vectorD[0] = -1e100;

	for (int i = 1; i < boundaryPoints; i++) {
		vectorA[i] = 0;
		vectorB[i] = -1;
		vectorC[i] = 0;
		vectorD[i] = -6.0 * turbProp.nu / (0.00708 * pow(turbProp.yCoordinate[i]+1, 2));
	}

	vectorA[turbProp.m - 1] = 0;
	vectorB[turbProp.m - 1] = -1;
	vectorD[turbProp.m - 1] = -1e100;

	for (int i = turbProp.m-boundaryPoints; i < turbProp.m-1; i++) {
		vectorA[i] = 0;
		vectorB[i] = -1;
		vectorC[i] = 0;
		vectorD[i] = -6.0 * turbProp.nu / (0.00708 * pow(1-turbProp.yCoordinate[i], 2));
	}

	for (int i = boundaryPoints; i < turbProp.m-boundaryPoints; i++) {
		vectorA[i] = 2.0 * turbProp.deltaTime / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*(turbProp.nu + sigma * (turbProp.nut[i] / 2.0 + turbProp.nut[i - 1] / 2.0)) / (turbProp.yCoordinate[i] - turbProp.yCoordinate[i - 1]);
		vectorB[i] = turbProp.deltaTime * (-alpha * turbProp.omega[i] - 2.0 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*((turbProp.nu + sigma * (turbProp.nut[i] / 2.0 + turbProp.nut[i - 1] / 2.0)) / (turbProp.yCoordinate[i] - turbProp.yCoordinate[i - 1]) + (turbProp.nu + sigma * (turbProp.nut[i] / 2.0 + turbProp.nut[i + 1] / 2.0)) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i]))) - 1;
		vectorC[i] = 2.0 * turbProp.deltaTime / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*(turbProp.nu + sigma * (turbProp.nut[i] / 2.0 + turbProp.nut[i + 1] / 2.0)) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i]);
		vectorD[i] = (-gamma * turbProp.betaML[i] * pow(((turbProp.xVelocity[i + 1] - turbProp.xVelocity[i - 1]) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])), 2.0))*turbProp.deltaTime - turbProp.omega[i]; // - max(0, sigmaD / omega[i] * ((k[i + 1] - k[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]))*((omega[i + 1] - omega[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]))))*deltaTime - omega[i];
	}

	thomasSolver(vectorA, vectorB, vectorC, vectorD, turbProp.omega);
}

void kSolver(caseProp& turbProp) {
	double sigmaStar = 0.5;
	double alphaStar = 0.09;

	vector<double> vectorA(turbProp.m, 0);
	vector<double> vectorB(turbProp.m, 0);
	vector<double> vectorC(turbProp.m, 0);
	vector<double> vectorD(turbProp.m, 0);

	vectorB[0] = -1.0;
	vectorC[0] = 0.0;
	vectorA[turbProp.m - 1] = 0;
	vectorB[turbProp.m - 1] = -1.0;
	vectorD[0] = 0;
	vectorD[turbProp.m - 1] = 0;

	for (int i = 1; i < turbProp.m - 1; i++){
		vectorA[i] = 2.0 * turbProp.deltaTime / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*(turbProp.nu + sigmaStar * (turbProp.nut[i] / 2 + turbProp.nut[i - 1] / 2)) / (turbProp.yCoordinate[i] - turbProp.yCoordinate[i - 1]);
		vectorB[i] = turbProp.deltaTime * (-alphaStar * turbProp.omega[i] - 2 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*((turbProp.nu + sigmaStar * (turbProp.nut[i] / 2 + turbProp.nut[i - 1] / 2)) / (turbProp.yCoordinate[i] - turbProp.yCoordinate[i - 1]) + (turbProp.nu + sigmaStar * (turbProp.nut[i] / 2 + turbProp.nut[i + 1] / 2)) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i]))) - 1;
		vectorC[i] = 2.0 * turbProp.deltaTime / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*(turbProp.nu + sigmaStar * (turbProp.nut[i] / 2 + turbProp.nut[i + 1] / 2)) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i]);
		vectorD[i] = (-turbProp.nut[i] * pow((turbProp.xVelocity[i + 1] - turbProp.xVelocity[i - 1]) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1]), 2.0))*turbProp.deltaTime - turbProp.k[i];
	}
	thomasSolver(vectorA, vectorB, vectorC, vectorD, turbProp.k);
}

void iterativeSolver(caseProp& turbProp){
	turbProp.xVelocity = turbProp.initXVelocity;
	vector<double> xVelocityTemp(turbProp.m);
	for (int i = 0; i < 10000; i++) {
		//cout << "#";
		xVelocityTemp = turbProp.xVelocity;
		omegaSolver(turbProp);
		kSolver(turbProp);
		flowSolver(turbProp);
		double res = 0;
		for (int i = 0; i < turbProp.m; i++) {
			turbProp.nut[i] = turbProp.k[i] / turbProp.omega[i];
			res = res + pow(turbProp.xVelocity[i] - xVelocityTemp[i], 2);
		}
		if (res < 1e-12){
			//cout << endl << "res: " << res << endl;
			break;
		}
		if (i==9999){
			cout << "did not converge" << endl;
			turbProp.convergence = 0;
		}
	}
}

void initialization(caseProp& turbProp){
	cout << "initialization residual:";
	iterativeSolver(turbProp);
	turbProp.initXVelocity = turbProp.xVelocity;
}