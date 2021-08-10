#pragma once

#include <algorithm>
#include <cmath>
//#include <accelmath.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <string.h>

#include <adolc/adolc.h>

#include "../dnsDataInterpreter/dnsDataInterpreterArray.h"

using namespace std;

void meshGen(int simpleGridingRatio, int m, double reTau, double yCoordinate[]);

template<typename T>
struct caseProp {
	int m, simpleGridingRatio, deltaTime;
	int convergence = 1;
	double nu, frictionVelocity, reTau;
	double* yCoordinate;
	double* dnsData;
	T* initXVelocity;
	T* xVelocity;
	T* k;
	T* omega;
	T* nut;
	T* betaML;
	T* R;
	T* solution;
};

template <typename T>
void caseInitialize(caseProp<T>& turbProp, double reTau){
	//initialize reTau number and update corresponding properties
	if (reTau == 180) {
		turbProp.reTau = reTau;
		turbProp.m = 199;
		turbProp.simpleGridingRatio = 80;
		turbProp.deltaTime = 1;
		turbProp.nu = 3.4e-4;
		turbProp.frictionVelocity = 6.37309e-2;
	}
	if (reTau == 550) {
		turbProp.reTau = reTau;
		turbProp.m = 199;
		turbProp.simpleGridingRatio = 300;
		turbProp.deltaTime = 1;
		turbProp.nu = 1e-4;
		turbProp.frictionVelocity = 5.43496e-2;
	}
	if (reTau == 1000) {
		turbProp.reTau = reTau;
		turbProp.m = 199;
		turbProp.simpleGridingRatio = 1500;
		turbProp.deltaTime = 1;
		turbProp.nu = 5e-5;
		turbProp.frictionVelocity = 5.00256e-2;
	}
	if (reTau == 2000) {
		turbProp.reTau = reTau;
		turbProp.m = 199;
		turbProp.simpleGridingRatio = 3000;
		turbProp.deltaTime = 1;
		turbProp.nu = 2.3e-5;
		turbProp.frictionVelocity = 4.58794e-2;
	}
	if (reTau == 5200) {
		turbProp.reTau = reTau;
		turbProp.m = 199;
		turbProp.simpleGridingRatio = 10000;
		turbProp.deltaTime = 1;
		turbProp.nu = 8e-6;
		turbProp.frictionVelocity =4.14872e-2;
	}
	//update mesh dimensions according to reTau
	turbProp.yCoordinate = new double[turbProp.m];
	turbProp.dnsData = new double[turbProp.m];
	turbProp.xVelocity = new T[turbProp.m];
	turbProp.initXVelocity = new T[turbProp.m];
	turbProp.k = new T[turbProp.m];
	turbProp.omega = new T[turbProp.m];
	turbProp.nut = new T[turbProp.m];
	turbProp.betaML = new T[turbProp.m];
	turbProp.R = new T[3*turbProp.m];
	turbProp.solution = new T[3*turbProp.m];

	for (int i = 0; i < turbProp.m; i++){
		turbProp.xVelocity[i] = 0;
		turbProp.initXVelocity[i] = 0;
		turbProp.k[i] = 1e-8;
		turbProp.omega[i] = 1e5;
		turbProp.nut[i] = 1e-5;
		turbProp.betaML[i] = 1;
	}

	for (int i = 0; i < 3*turbProp.m; i++){
		turbProp.R[i] = 1;
		turbProp.solution[i] = 0;
	}
	//update mesh data on yCoordinate
	meshGen(turbProp.simpleGridingRatio, turbProp.m, reTau, turbProp.yCoordinate);

	//update interpreted DNS data
	dnsDataInterpreter dnsDataTable(turbProp.yCoordinate, reTau, turbProp.m);
    memcpy(turbProp.dnsData, dnsDataTable.U, sizeof(double)*turbProp.m);
}

template<typename T>
void thomasSolver(T vectorA[], T vectorB[], T vectorC[], T vectorD[], T solution[], int vectorSize){
	T newVectorC[vectorSize];
	T newVectorD[vectorSize];

	for (int i = 0; i < vectorSize; i ++){
		newVectorC[i] = 0;
		newVectorD[i] = 0;
	}

	newVectorC[0] = vectorC[0] / vectorB[0];
	newVectorD[0] = vectorD[0] / vectorB[0];

	//#pragma acc kernels
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

template<typename T>
void linearDiscretization(caseProp<T>& turbProp, T vectorA[], T vectorB[], T vectorC[], T vectorD[]){
	for (int i = 0; i < turbProp.m; i++){
		turbProp.nut[i] = turbProp.k[i]/turbProp.omega[i];
	}
	//discretizatin of flow equation
	vectorB[0] = -1;
	vectorC[0] = 0;
	vectorA[turbProp.m - 1] = 0;
	vectorB[turbProp.m - 1] = -1;
	vectorD[0] = 0;
	vectorD[turbProp.m - 1] = 0;

	//#pragma acc parallel loop
	for (int i = 1; i < turbProp.m - 1; i++) {
		vectorA[i] = 2.0 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*(turbProp.nu + turbProp.nut[i - 1] / 2.0 + turbProp.nut[i] / 2.0) / (turbProp.yCoordinate[i] - turbProp.yCoordinate[i - 1]);
		vectorB[i] = -2.0 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*((turbProp.nu + turbProp.nut[i] / 2.0 + turbProp.nut[i + 1] / 2.0) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i]) + (turbProp.nu + turbProp.nut[i - 1] / 2.0 + turbProp.nut[i] / 2.0) / (turbProp.yCoordinate[i] - turbProp.yCoordinate[i - 1])) - 1/turbProp.deltaTime;
		vectorC[i] = 2.0 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*(turbProp.nu + turbProp.nut[i] / 2.0 + turbProp.nut[i + 1] / 2.0) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i]);
		vectorD[i] = -pow(turbProp.frictionVelocity, 2) - turbProp.xVelocity[i]/turbProp.deltaTime;
	}

	//discretizatin of omega equation
	double sigma = 0.5;
	double alpha = 3.0 / 40.0;
	double gamma = 5.0 / 9.0;
	int boundaryPoints = 3;

	vectorB[turbProp.m] = -1;
	vectorC[turbProp.m] = 0;
	vectorD[turbProp.m] = -1e100;

	for (int i = 1; i < boundaryPoints; i++) {
		vectorA[i+turbProp.m] = 0;
		vectorB[i+turbProp.m] = -1;
		vectorC[i+turbProp.m] = 0;
		vectorD[i+turbProp.m] = -6.0 * turbProp.nu / (0.00708 * pow(turbProp.yCoordinate[i]+1, 2));
	}

	vectorA[2*turbProp.m-1] = 1;
	vectorB[2*turbProp.m-1] = -1;
	vectorD[2*turbProp.m-1] = -1e100;

	for (int i = turbProp.m-boundaryPoints; i < turbProp.m-1; i++) {
		vectorA[i+turbProp.m] = 0;
		vectorB[i+turbProp.m] = -1;
		vectorC[i+turbProp.m] = 0;
		vectorD[i+turbProp.m] = -6.0 * turbProp.nu / (0.00708 * pow(1-turbProp.yCoordinate[i], 2));
	}

	//#pragma acc parallel loop
	for (int i = boundaryPoints; i < turbProp.m-boundaryPoints; i++) {
		vectorA[i+turbProp.m] = 2.0 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*(turbProp.nu + sigma * (turbProp.nut[i] / 2.0 + turbProp.nut[i - 1] / 2.0)) / (turbProp.yCoordinate[i] - turbProp.yCoordinate[i - 1]);
		vectorB[i+turbProp.m] = -alpha * turbProp.omega[i] - 2.0 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*((turbProp.nu + sigma * (turbProp.nut[i] / 2.0 + turbProp.nut[i - 1] / 2.0)) / (turbProp.yCoordinate[i] - turbProp.yCoordinate[i - 1]) + (turbProp.nu + sigma * (turbProp.nut[i] / 2.0 + turbProp.nut[i + 1] / 2.0)) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i])) - 1/turbProp.deltaTime;
		vectorC[i+turbProp.m] = 2.0 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*(turbProp.nu + sigma * (turbProp.nut[i] / 2.0 + turbProp.nut[i + 1] / 2.0)) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i]);
		vectorD[i+turbProp.m] = -gamma * turbProp.betaML[i] * pow(((turbProp.xVelocity[i + 1] - turbProp.xVelocity[i - 1]) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])), 2.0) - turbProp.omega[i]/turbProp.deltaTime; // - max(0, sigmaD / omega[i] * ((k[i + 1] - k[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]))*((omega[i + 1] - omega[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]))))*deltaTime - omega[i];
	}


	//discretizatin of k equation
	double sigmaStar = 0.5;
	double alphaStar = 0.09;

	vectorB[2*turbProp.m] = -1.0;
	vectorC[2*turbProp.m] = 0.0;
	vectorA[3*turbProp.m - 1] = 0;
	vectorB[3*turbProp.m - 1] = -1.0;
	vectorD[2*turbProp.m] = 0;
	vectorD[3*turbProp.m - 1] = 0;

	//#pragma acc parallel loop
	for (int i = 1; i < turbProp.m - 1; i++){
		vectorA[i+2*turbProp.m] = 2.0 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*(turbProp.nu + sigmaStar * (turbProp.nut[i] / 2 + turbProp.nut[i - 1] / 2)) / (turbProp.yCoordinate[i] - turbProp.yCoordinate[i - 1]);
		vectorB[i+2*turbProp.m] = -alphaStar * turbProp.omega[i] - 2 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*((turbProp.nu + sigmaStar * (turbProp.nut[i] / 2 + turbProp.nut[i - 1] / 2)) / (turbProp.yCoordinate[i] - turbProp.yCoordinate[i - 1]) + (turbProp.nu + sigmaStar * (turbProp.nut[i] / 2 + turbProp.nut[i + 1] / 2)) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i])) - 1/turbProp.deltaTime;
		vectorC[i+2*turbProp.m] = 2.0 / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])*(turbProp.nu + sigmaStar * (turbProp.nut[i] / 2 + turbProp.nut[i + 1] / 2)) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i]);
		vectorD[i+2*turbProp.m] = -turbProp.nut[i] * pow((turbProp.xVelocity[i + 1] - turbProp.xVelocity[i - 1]) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1]), 2.0) - turbProp.k[i]/turbProp.deltaTime;
	}
}

template<typename T>
void flowSolver(caseProp<T>& turbProp) {
	//initiate linear system
	T vectorA[3*turbProp.m];
	T vectorB[3*turbProp.m];
	T vectorC[3*turbProp.m];
	T vectorD[3*turbProp.m];

	for (int i = 0; i < 3*turbProp.m; i ++){
		vectorA[i] = 0;
		vectorB[i] = 0;
		vectorC[i] = 0;
		vectorD[i] = 0;
	}
	linearDiscretization(turbProp, vectorA, vectorB, vectorC, vectorD);
	thomasSolver(vectorA, vectorB, vectorC, vectorD, turbProp.solution, 3*turbProp.m);

	for (int i = 0; i < turbProp.m; i ++){
		turbProp.xVelocity[i] = turbProp.solution[i];
		turbProp.omega[i] = turbProp.solution[turbProp.m + i];
		turbProp.k[i] = turbProp.solution[2*turbProp.m + i];
	}

};

template<typename T>
void residualUpdate(caseProp<T>& turbProp){
	//initiate linear system
	T vectorA[3*turbProp.m];
	T vectorB[3*turbProp.m];
	T vectorC[3*turbProp.m];
	T vectorD[3*turbProp.m];

	for (int i = 0; i < 3*turbProp.m; i ++){
		vectorA[i] = 0;
		vectorB[i] = 0;
		vectorC[i] = 0;
		vectorD[i] = 0;
	}

	linearDiscretization(turbProp, vectorA, vectorB, vectorC, vectorD);

	turbProp.R[0] = 0;
	for (int i = 1; i < turbProp.m-1; i++){
		turbProp.R[i] = vectorA[i]*turbProp.xVelocity[i-1]+vectorB[i]*turbProp.xVelocity[i]+vectorC[i]*turbProp.xVelocity[i+1]-vectorD[i];
	}
	turbProp.R[turbProp.m-1] = 0;
	turbProp.R[turbProp.m] = 0;
	for (int i = 1; i < turbProp.m-1; i++){
		turbProp.R[i+turbProp.m] = vectorA[i+turbProp.m]*turbProp.omega[i-1]+vectorB[i+turbProp.m]*turbProp.omega[i]+vectorC[i+turbProp.m]*turbProp.omega[i+1]-vectorD[i+turbProp.m];
	}
	turbProp.R[2*turbProp.m-1] = 0;
	turbProp.R[2*turbProp.m] = 0;
	for (int i = 1; i < turbProp.m-1; i++){
		turbProp.R[i+2*turbProp.m] = vectorA[i+2*turbProp.m]*turbProp.k[i-1]+vectorB[i+2*turbProp.m]*turbProp.k[i]+vectorC[i+2*turbProp.m]*turbProp.k[i+1]-vectorD[i+2*turbProp.m];
	}
	turbProp.R[3*turbProp.m-1] = 0;
}


inline void iterativeSolver(caseProp<double>& turbProp){
    for (int i = 0; i < 199; i++){
        turbProp.xVelocity[i] = turbProp.initXVelocity[i];
    }
    for (int i = 0; i < 10000; i++) {
        flowSolver(turbProp);
        residualUpdate(turbProp);

        double maxElem = *max_element(turbProp.R, turbProp.R+turbProp.m);
        double minElem = *min_element(turbProp.R, turbProp.R+turbProp.m);
        double res = max(abs(maxElem), abs(minElem));

        if ( res < 1e-4 && i > 20){
            break;
        }
        if (i==9999){
            std::cout << "did not converge" << std::endl;
            turbProp.convergence = 0;
        }
    }
}

inline void iterativeSolver(caseProp<adouble>& turbProp){
    for (int i = 0; i < 199; i++){
        turbProp.xVelocity[i] = 0;
    }
    for (int i = 0; i < 10000; i++) {
        flowSolver(turbProp);
        residualUpdate(turbProp);

        vector<double> rArray(3*turbProp.m, 1);
        for (int j = 0; j < 3*turbProp.m; j++){
            rArray[j] = turbProp.R[j].value();
        }

        double maxElem = *max_element(rArray.begin(), rArray.end());
        double minElem = *min_element(rArray.begin(), rArray.end());
        double res = max(abs(maxElem), abs(minElem));
        if ( res < 1e-6 && i > 1000){
            break;
        }
        if (i==9999){
            std::cout << "did not converge" << std::endl;
            turbProp.convergence = 0;
        }
    }
}

template<typename T>
void initialization(caseProp<T>& turbProp){
	iterativeSolver(turbProp);
	memcpy(turbProp.initXVelocity, turbProp.xVelocity, sizeof(double)*turbProp.m);
}

