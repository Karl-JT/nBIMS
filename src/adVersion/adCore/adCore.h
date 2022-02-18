#pragma once

#include <adolc/adolc.h>
#include <ChannelCppSolverBiMesh.h>

double adIterativeSolver(caseProp<adouble>& turbProp, double approximateCoefProposal[], double gradientVector[], double hessianMatrix[], int numCoef);
