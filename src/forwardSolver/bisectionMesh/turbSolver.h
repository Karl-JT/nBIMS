#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <string.h>

#include "ChannelCppSolverBiMesh.h"

class forwardSolver : public caseProp {
public:
	int procid = 0;

	forwardSolver(int level);
	~forwardSolver(){};

	void updateParameter(double sample[], int numCoef);
	void updateReference(string filePath);

	void soluFwd(); 
	void updateInitialState(){};
	double lnLikelihood(double sample[], int numCoef);	
	double adSolver(double sampleProposal[], double gradientVector[], double hessianMatrix[], int numCoef);
};
