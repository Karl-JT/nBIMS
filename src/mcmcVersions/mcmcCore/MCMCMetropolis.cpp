#include "MCMCMetropolis.h"
#include <time.h>

covProposal::covProposal(int numCoef_, int procid_) : numCoef(numCoef_), procid(procid_){
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);

	gsl_rng_set(r, time(NULL));

	priorCovMatrix = gsl_matrix_alloc(numCoef, numCoef);     
	proposalCovMatrix = gsl_matrix_alloc(numCoef, numCoef); 	
	xVector = gsl_matrix_alloc(numCoef, 1);			
	lMatrix = gsl_matrix_alloc(numCoef, numCoef);			
	precisionMatrix = gsl_matrix_alloc(numCoef, numCoef);	

	proposalVector = gsl_vector_alloc(numCoef);	
	proposalStep = gsl_vector_alloc(numCoef);	
	muVector = gsl_vector_alloc(numCoef);
	zeroVector = gsl_vector_alloc(numCoef);
	gsl_vector_set_zero(zeroVector);

	acceptanceRate = 0.5;
	scale = 1;
}

void covProposal::initMuVector(vector<double> intMuVector){
	for (int i = 0; i < numCoef; i ++){
		gsl_matrix_set(xVector, i, 0, intMuVector[i]);
		gsl_vector_set(muVector, i, intMuVector[i]);
	}
}

void covProposal::initCovMatrixAsMatern(double spatioScale, double varScale, vector<double> yCoordinate){
	double cij = 0;
	double dij = 0;
	for (int i = 0; i < numCoef; i++){
		for (int j = 0; j < numCoef; j++){
			dij = abs(yCoordinate[i] - yCoordinate[j]);
			cij = pow(varScale, 2)*(1 + sqrt(5)*dij/spatioScale + 5*pow(dij, 2)/(3*pow(spatioScale, 2)))*exp(-sqrt(5)*dij/spatioScale);
			gsl_matrix_set(priorCovMatrix, i, j, cij);
		}
	}

	gsl_matrix_memcpy(lMatrix, priorCovMatrix);
	gsl_linalg_cholesky_decomp(lMatrix);

	gsl_matrix_memcpy(precisionMatrix, priorCovMatrix);
	gsl_linalg_cholesky_decomp(precisionMatrix);
	gsl_linalg_cholesky_invert(precisionMatrix);

	gsl_matrix_memcpy(proposalCovMatrix, priorCovMatrix);
}

void covProposal::initCovMatrixAsIdentity(double stepSize = 0.5){
	gsl_matrix_set_identity(priorCovMatrix);
	gsl_matrix_scale(priorCovMatrix, stepSize);
	gsl_matrix_memcpy(lMatrix, priorCovMatrix);
	gsl_linalg_cholesky_decomp(lMatrix);

	gsl_matrix_memcpy(precisionMatrix, priorCovMatrix);
	gsl_linalg_cholesky_decomp(precisionMatrix);
	gsl_linalg_cholesky_invert(precisionMatrix);

	gsl_matrix_memcpy(proposalCovMatrix, priorCovMatrix);
}

void covProposal::muVectorUpdate(vector<double> updateVector){
	for (int i = 0; i < numCoef; i++){
		gsl_vector_set(muVector, i, updateVector[i]);
	}
}

vector<double> covProposal::muVectorReturn(){
	vector<double> returnVector(numCoef);
	for (int i = 0; i < numCoef; i++){
		returnVector[i] = gsl_vector_get(muVector, i);
	}
	return returnVector;
}

vector<double> covProposal::pCNProposalGenerate(){
	vector<double> returnProposal(numCoef);
	gsl_ran_multivariate_gaussian(r, zeroVector, lMatrix, proposalStep);
	for (int i = 0; i < numCoef; i++){
		returnProposal[i] = sqrt(1-pow(scale, 2))*gsl_vector_get(muVector, i) + scale*gsl_vector_get(proposalStep, i);
		gsl_vector_set(proposalVector, i, returnProposal[i]);
	}
	return returnProposal;
}

void covProposal::RWProposalGenerate(double returnProposal[]){
	gsl_ran_multivariate_gaussian(r, zeroVector, lMatrix, proposalStep);
	for (int i = 0; i < numCoef; i++){
		returnProposal[i] = gsl_vector_get(proposalStep, i);
	}
}

double covProposal::returnPrior(){
	gsl_vector* priorVector1 = gsl_vector_alloc(numCoef);
	gsl_vector* priorVector2 = gsl_vector_alloc(numCoef);

	double result;

	gsl_vector_memcpy(priorVector1, proposalVector);
	gsl_vector_add_constant(priorVector1, -1.0);
	gsl_blas_dgemv(CblasNoTrans, 1, precisionMatrix, priorVector1, 0, priorVector2);
	gsl_blas_ddot(priorVector1, priorVector2, &result);
	cout << "result: " << result;
	return result;
}

void covProposal::updateAcceptanceRate(double alpha){
	acceptanceRate = acceptanceRate*0.99+0.01*pow(M_E, alpha);
	cout << "acceptanceRate: " << acceptanceRate << endl;
}

void covProposal::updatePreMatrix(double hessian[]){
	for (int i = 0; i < numCoef; i++){
		for (int j = 0; j < numCoef; j++){
			gsl_matrix_set(precisionMatrix, i, j, hessian[i*numCoef+j]);
		}
	}

	gsl_matrix_memcpy(priorCovMatrix, precisionMatrix);
	gsl_linalg_cholesky_decomp(priorCovMatrix);
	gsl_linalg_cholesky_invert(priorCovMatrix);

	gsl_matrix_memcpy(lMatrix, priorCovMatrix);
	gsl_linalg_cholesky_decomp(lMatrix);
};


void covProposal::adaptCovMatrix(){}


compUniformProposal::compUniformProposal(int numCoef_, int procid_, double randomSeed) : numCoef(numCoef_), procid(procid_){
	fourierCoef.resize(numCoef, 0);
	acceptanceRate = 0.;

	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);

	gsl_rng_set(r, randomSeed);
}

double compUniformProposal::compUniformProposalReturn(int idx){
	return fourierCoef[idx];
}

vector<double> compUniformProposal::compUniformProposalGenerator(){
	for (int i = 0; i < numCoef; i++){
		fourierCoef[i] = gsl_rng_uniform(r)*2 - 1;
	}
	return fourierCoef;
}

void compUniformProposal::compGaussianProposalGenerator(double approximateCoefProposal[]){
	for (int i = 0; i < numCoef; i++){
		approximateCoefProposal[i] = gsl_ran_gaussian(r, scale);
	}
}

void compUniformProposal::updateAcceptanceRate(double alpha){
	acceptanceRate = counter*acceptanceRate/(counter + 1) + exp(alpha)/(counter+1);
	counter++;
}

void compUniformProposal::clearAcceptanceRate(){
	counter = 0;
	acceptanceRate = 0;
}

compGaussianProposal::compGaussianProposal(int numCoef_, int procid_) : numCoef(numCoef_), procid(procid_){
	fourierCoef.resize(numCoef, 0);
	fourierCoef[0] = 1;
	scale.resize(numCoef, 0.1);
	acceptanceRate.resize(numCoef, 0.35);

	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);

	gsl_rng_set(r, procid*31+5);
}

double compGaussianProposal::compGaussianProposalReturn(int idx){
	return fourierCoef[idx];
}

double compGaussianProposal::compGaussianProposalGenerator(int idx, double coef){
	coef = coef + gsl_ran_gaussian(r, scale[idx]);
	return coef;
}

void compGaussianProposal::updateAcceptanceRate(int idx, double alpha, int j){
	acceptanceRate[idx] = acceptanceRate[idx]*0.999+0.001*pow(M_E, alpha);
	if (acceptanceRate[idx] < 0.2 && j <20000){
		scale[idx] = scale[idx]*0.5;
		acceptanceRate[idx] = 0.3;
	} else if (acceptanceRate[idx] > 0.5 && j < 20000){
		scale[idx] = scale[idx]*2;
		acceptanceRate[idx] = 0.4;
	}
	cout << "acceptanceRate: ";
	for (int i = 0; i < numCoef; i++){
		cout << acceptanceRate[i] << " "; 
	}
	cout << endl << "scale: ";
	for (int i = 0; i < numCoef; i++){
		cout << scale[i] << " ";
	}
}

chainIO::chainIO(string outputPath, int procid_, int numCoef_) : procid(procid_), numCoef(numCoef_){

	yFilePath = outputPath + "/yFile" + to_string(procid) + ".csv";
	uFilePath = outputPath + "/uFile" + to_string(procid) + ".csv";
	betaFilePath = outputPath + "/betaFile" + to_string(procid) + ".csv";
	MCMCFilePath = outputPath + "/MCMCChain" + to_string(procid) + ".csv";
	costFilePath = outputPath + "/costFile" + to_string(procid) + ".csv";
	likelihoodFilePath = outputPath + "/likelihoodFile" + to_string(procid) + ".csv";
	A1FilePath = outputPath + "/A1File" + to_string(procid) + ".csv";
	A2FilePath = outputPath + "/A2File" + to_string(procid) + ".csv";
	A3FilePath = outputPath + "/A3File" + to_string(procid) + ".csv";
	A4FilePath = outputPath + "/A4File" + to_string(procid) + ".csv";
	A5FilePath = outputPath + "/A5File" + to_string(procid) + ".csv";
	A6FilePath = outputPath + "/A6File" + to_string(procid) + ".csv";
	A7FilePath = outputPath + "/A7File" + to_string(procid) + ".csv";
	A8FilePath = outputPath + "/A8File" + to_string(procid) + ".csv";	
	QFilePath = outputPath + "/QFile" + to_string(procid) + ".csv";
}

void chainIO::chainReadCoef(vector<double> coef){
	fin.open(MCMCFilePath.c_str());
    fin.seekg(-2,ios_base::end);                // go to one spot before the EOF

    bool keepLooping = true;
    while(keepLooping) {
        char ch;
        fin.get(ch);                            // Get current byte's data

        if((int)fin.tellg() <= 1) {             // If the data was at or before the 0th byte
            fin.seekg(0);                       // The first line is the last line
            keepLooping = false;                // So stop there
        }
        else if(ch == '\n') {                   // If the data was a newline
            keepLooping = false;                // Stop at the current position.
        }
        else {                                  // If the data was neither a newline nor at the 0 byte
            fin.seekg(-2,ios_base::cur);        // Move to the front of that data, then to the front of the data before it
		}
	}
    string lastLine;            
    getline(fin,lastLine);                      // Read the current line
	stringstream s(lastLine);

    int i = 0;
	while (getline(s, wordBuff, ' ')){
		numBuff = stof(wordBuff);
		coef[i] = numBuff;
		i ++;
	}

    fin.close();
}

void chainIO::chainWriteCoef(double coef[]){
	MCMCChain.open(MCMCFilePath.c_str(), ios_base::app);
	for (int i = 0; i < numCoef; i++){
		MCMCChain << coef[i] << " ";
	}
	MCMCChain << endl;
	MCMCChain.close();
}

void chainIO::chainWriteCost(double costValue){
	costFunction.open(costFilePath.c_str(), ios_base::app);
	costFunction << costValue << endl;
	costFunction.close();
}

void chainIO::chainWriteBeta(double beta[], int arraySize){
	betaFile.open(betaFilePath.c_str(), ios_base::app | ios_base::out);
	for (int i = 0; i < arraySize; i++) {
		betaFile << beta[i] << " ";
	}
	betaFile << endl;
	betaFile.close();
}

void chainIO::chainWriteCoordinate(double coordinate[], int arraySize){
	yFile.open(yFilePath.c_str(), ios_base::app);
	for (int i = 0; i < arraySize; i++) {
		yFile << coordinate[i] << " ";
	}
	yFile.close();
}

void chainIO::chainWriteVelocity(double velocity[], int arraySize){
	uFile.open(uFilePath.c_str(), ios_base::app);
	for (int i = 0; i < arraySize; i++) {
		uFile << velocity[i] << " ";
	}
	uFile << endl;
	uFile.close();
}

void chainIO::chainWriteLikelihood(double likelihood){
	likelihoodFile.open(likelihoodFilePath.c_str(), ios_base::app);
	likelihoodFile << likelihood << endl;
	likelihoodFile.close();
}

void chainIO::chainWriteA1(double A1){
	A1File.open(A1FilePath.c_str(), ios_base::app);
	A1File << A1 << endl;
	A1File.close();
}

void chainIO::chainWriteA2(double A2){
	A2File.open(A2FilePath.c_str(), ios_base::app);
	A2File << A2 << endl;
	A2File.close();
}

void chainIO::chainWriteA3(double A3){
	A3File.open(A3FilePath.c_str(), ios_base::app);
	A3File << A3 << endl;
	A3File.close();
}

void chainIO::chainWriteA4(double A4){
	A4File.open(A4FilePath.c_str(), ios_base::app);
	A4File << A4 << endl;
	A4File.close();
}

void chainIO::chainWriteA5(double A5){
	A5File.open(A5FilePath.c_str(), ios_base::app);
	A5File << A5 << endl;
	A5File.close();
}

void chainIO::chainWriteA6(double A6){
	A6File.open(A6FilePath.c_str(), ios_base::app);
	A6File << A6 << endl;
	A6File.close();
}

void chainIO::chainWriteA7(double A7){
	A7File.open(A7FilePath.c_str(), ios_base::app);
	A7File << A7 << endl;
	A7File.close();
}

void chainIO::chainWriteA8(double A8){
	A8File.open(A8FilePath.c_str(), ios_base::app);
	A8File << A8 << endl;
	A8File.close();
}

void chainIO::chainWriteQ(double Q){
	QFile.open(QFilePath.c_str(), ios_base::app);
	QFile << Q << endl;
	QFile.close();
}

