#pragma once

#include <dolfin.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <string.h>
#include <random>

#include "ChannelCppSolverBiMesh.h"

using namespace dolfin;

class channelFlowAdjoint : public caseProp {
public:
	Mat PRPWMat, PRPBetaMat;
	Vec PJPWVec, adjoints, adjointTerm, PJPBetaVec;
	PetscMPIInt size;
	PetscInt restart;
	KSP ksp; PC pc;

	Mat obsMat;
	Vec inputSpace, workSpace1, workSpace2;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution{0.0, 1.0};

	//supportive methods
	PetscErrorCode MatMult_Shell(Mat Action, Vec input, Vec output);
	PetscErrorCode MatGetDiagonal_Shell(Mat Action, Vec diag);

	channelFlowAdjoint(int level);
	~channelFlowAdjoint(){};

	void updateParameter(double sample[], int numCoef);

	void soluFwd(); 
	double lnLikelihood(double sample[], int numCoef);	
	double adSolver();
};
