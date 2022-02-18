#pragma once

// #include <algorithm>
// #include <cmath>
// #include <fstream>
#include <iostream>
// #include <sstream>
// #include <type_traits>
#include <string.h>
#include <memory>

#include <adolc/adolc.h>
#include <dnsDataInterpreterArray.h>

class caseProp {
public:
	int n0, m, l, simpleGridingRatio, deltaTime;
	int convergence = 1;
	double nu, frictionVelocity, reTau, cut_off;
	double boundaryVelocity[2] = {0};
	std::unique_ptr<double[]> yCoordinate;
	std::unique_ptr<double[]> dnsData;
	std::unique_ptr<double[]> nodeWeight;
	std::unique_ptr<double[]> initXVelocity;
	std::unique_ptr<double[]> pressure;
	std::unique_ptr<adouble[]> xVelocity;
	std::unique_ptr<adouble[]> k;
	std::unique_ptr<adouble[]> omega;
	std::unique_ptr<adouble[]> nut;
	std::unique_ptr<adouble[]> betaML;
	std::unique_ptr<adouble[]> R;
	std::unique_ptr<adouble[]> solution;
	std::unique_ptr<adouble[]> xVelocityGradient;

	double validationYCoordinate[289];
	double validationVelocity[289];

	caseProp(){};
	~caseProp(){};
	
	void meshGen(int l, int n0, double yCoordinate[]);

	void caseInitialize(int l_, double reTau = 550);

	void thomasSolver(adouble vectorA[], adouble vectorB[], adouble vectorC[], adouble vectorD[], adouble solution[], int vectorSize);

	void linearDiscretization(adouble vectorA[], adouble vectorB[], adouble vectorC[], adouble vectorD[], int stepCount);

	void flowSolver(int stepCount);

	void residualUpdate();

	void iterativeSolver();

	void initialization();

	void updateBoundaryVelocity(double velocity[]);

	void updatePressure(double inputPressure[]);
};
