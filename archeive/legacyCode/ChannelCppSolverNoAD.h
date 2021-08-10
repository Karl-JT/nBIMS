#ifndef CHANNELCPPSOLVERNOAD
#define CHANNELCPPSOLVERNOAD

#include "dnsDataInterpreter.h"
#include <vector>

using namespace std;

void meshGen(int simpleGridingRatio, int m, double reTau, vector<double> & yCoordinate);

class caseProp {
public:
	int m, simpleGridingRatio, deltaTime;
	int convergence = 1;
	double nu, frictionVelocity;
	vector<double> yCoordinate;
	vector<double> xVelocity;
	vector<double> initXVelocity;
	vector<double> dnsData;
	vector<double> k;
	vector<double> omega;
	vector<double> nut;
	vector<double> betaML;


	caseProp(double);
	~caseProp(){};
};

void thomasSolver(vector<double>& vectorA, vector<double>& vectorB, vector<double>& vectorC, vector<double>& vectorD, vector<double>& solution);

void flowSolver(caseProp& turbProp);

void omegaSolver(caseProp& turbProp);

void kSolver(caseProp& turbProp);

void iterativeSolver(caseProp& turbProp);

void initialization(caseProp& turbProp);

#endif