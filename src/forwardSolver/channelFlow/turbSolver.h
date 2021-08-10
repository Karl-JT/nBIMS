#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <string.h>

#include "ChannelCppSolverBiMesh.h"

class reducedSolver : public caseProp {
public:
	reducedSolver(int level);
	~reducedSolver(){};

	void updateParameter(double sample[], int numCoef);

	void soluFwd(); 
	double lnLikelihood(double sample[], int numCoef);	
	double adSolver(double sampleProposal[], double gradientVector[], double hessianMatrix[], int numCoef);
};
