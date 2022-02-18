#pragma once

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <random>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <time.h>

//#include "../../forwardSolver/bisectionMesh/ChannelCppSolverBiMesh.h" 

using namespace std;

class covProposal{
private:
	const int numCoef; 
	int procid;
	gsl_rng *r;
	const gsl_rng_type *T;
	gsl_matrix* priorCovMatrix;
	gsl_matrix* proposalCovMatrix;      	
	gsl_matrix* xVector;			
	gsl_matrix* lMatrix;			
	gsl_matrix* precisionMatrix;	
	gsl_vector* zeroVector;
	gsl_vector* proposalVector;	
	gsl_vector* proposalStep;	
	gsl_vector* muVector;			
public:
	double scale;
	double acceptanceRate;

	covProposal(int numCoef_, int procid_ = 0);
	~covProposal(){};	

	void initMuVector(vector<double> initMuVector);
	void initCovMatrix(double spatioScale, double varScale, vector<double> yCoordiante);
	void initCovMatrixAsMatern(double spatioScale, double varScale, vector<double> yCoordinate);
	void initCovMatrixAsIdentity(double stepSize);
	void updatePreMatrix(double hessian[]);
	void adaptCovMatrix();
	void muVectorUpdate(vector<double> updateVector);
	void updateAcceptanceRate(double alpha);

	vector<double> muVectorReturn();
	vector<double> pCNProposalGenerate();
	void RWProposalGenerate(double returnProposal[]);

	double returnPrior();
};

class compUniformProposal{
private:
	int numCoef;
	int procid;
	int counter = 0;

	gsl_rng *r;
	const gsl_rng_type *T;

	vector<double> fourierCoef;

public:
	double scale = 1;
	double acceptanceRate;

	compUniformProposal(int numCoef_, int procid_, double randomSeed);
	~compUniformProposal(){};

	double compUniformProposalReturn(int idx);
	vector<double> compUniformProposalGenerator();
	void compGaussianProposalGenerator(double approximateCoefProposal[]);

	void updateAcceptanceRate(double alpha);
	void clearAcceptanceRate();
};

class compGaussianProposal{
private:
	int numCoef;
	int procid;
	vector<double> fourierCoef;
	vector<double> scale;
	vector<double> acceptanceRate;

	gsl_rng *r;
	const gsl_rng_type *T;

public:
	compGaussianProposal(int numCoef_, int procid_);
	~compGaussianProposal(){};

	double compGaussianProposalReturn(int idx);
	double compGaussianProposalGenerator(int idx, double coef);

	void updateAcceptanceRate(int idx, double alpha, int j);
};

class chainIO{
private:
	int procid;
	int numCoef;
	double numBuff;
	string wordBuff;
	string filename;
	string yFilePath; 
	string uFilePath;
	string betaFilePath;
	string MCMCFilePath;
	string costFilePath;
	string likelihoodFilePath;
	string A1FilePath, A2FilePath, A3FilePath, A4FilePath, A5FilePath, A6FilePath, A7FilePath, A8FilePath, QFilePath;

	ifstream fin;

	ofstream MCMCChain;
	ofstream costFunction;
	ofstream yFile;
	ofstream uFile;
	ofstream betaFile;
	ofstream likelihoodFile;
	ofstream A1File;
	ofstream A2File;
	ofstream A3File;
	ofstream A4File;
	ofstream A5File;
	ofstream A6File;
	ofstream A7File;
	ofstream A8File;
	ofstream QFile;

public:
	chainIO(string outputPath, int procid_, int numCoef_);
	~chainIO(){};

	void chainReadCoef(vector<double> coef);

	void chainWriteCoef(double coef[]);
	void chainWriteCost(double costValue);
	void chainWriteBeta(double beta[], int arraySize);
	void chainWriteCoordinate(double coordinate[], int arraySize);
	void chainWriteVelocity(double velocity[], int arraySize);
	void chainWriteLikelihood(double likelihood);
	void chainWriteA1(double A1);
	void chainWriteA2(double A2);
	void chainWriteA3(double A3);
	void chainWriteA4(double A4);
	void chainWriteA5(double A5);
	void chainWriteA6(double A6);
	void chainWriteA7(double A7);
	void chainWriteA8(double A8);
	void chainWriteQ(double Q);
};
