#include "MCMCMetropolis.h"
#include "MCMCChain.h"
#include "linearAlgebra.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <random>

template <typename chainType>
class stochasticNewtonMCMCChain : public MCMCChain<chainType> {
private:
	gsl_rng *r;
	const gsl_rng_type *T;

public:
	std::unique_ptr<double[]> newtonStepArray;
	std::unique_ptr<double[]> gradientVectorM;
	std::unique_ptr<double[]> gradientVectorY;
	std::unique_ptr<double[]> hessianMatrixM;
	std::unique_ptr<double[]> hessianMatrixY;

	int RWSwitch = 1;

	std::shared_ptr<covProposal> sampleGenerator;

	stochasticNewtonMCMCChain(string outputPath, int MLl_, int procid_, int numCoef_, int samplerLevel_);
	~stochasticNewtonMCMCChain(){};

	void startPoint(double QoI[]);
	void runStep();
	virtual void run(double QoI[]);
	double alphaUpdate(double lnLikelihoodt0, double lnLikelihoodt1, double vectorM[], double vectorY[], double gradientVectorM[], double gradientVectorY[], double hessianMatrixM[], double hessianMatrixY[]);
};

template <typename chainType>
stochasticNewtonMCMCChain<chainType>::stochasticNewtonMCMCChain(string outputPath, int MLl_, int procid_, int numCoef_, int samplerLevel_) : MCMCChain<chainType>(outputPath, MLl_, procid_, numCoef_, samplerLevel_){
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	gsl_rng_set(r, 2019);

    sampleGenerator = std::make_shared<covProposal>(this->numCoef, 0);
	sampleGenerator->scale = 1;

	this->MCMCChainIO = std::make_shared<chainIO>(outputPath, 0, this->numCoef); 

	newtonStepArray = std::make_unique<double[]>(this->numCoef);
	gradientVectorM = std::make_unique<double[]>(this->numCoef);
	gradientVectorY = std::make_unique<double[]>(this->numCoef);
	hessianMatrixM = std::make_unique<double[]>(this->numCoef*this->numCoef);
	hessianMatrixY = std::make_unique<double[]>(this->numCoef*this->numCoef);
}

template <typename chainType>
double stochasticNewtonMCMCChain<chainType>::alphaUpdate(double lnLikelihoodt0, double lnLikelihoodt1, double vectorM[], double vectorY[], double gradientVectorM[], double gradientVectorY[], double hessianMatrixM[], double hessianMatrixY[]){
	double alpha;
	double piY;
	double piM;
	double detHM;
	double detHY;
	double workSpaceScale;

	int s;
	gsl_permutation* p = gsl_permutation_alloc(this->numCoef);
	gsl_vector* workSpaceVector = gsl_vector_alloc(this->numCoef);
	gsl_matrix* workSpaceMatrix = gsl_matrix_alloc(this->numCoef, this->numCoef);
	gsl_vector* M = gsl_vector_alloc(this->numCoef);
	gsl_vector* Y = gsl_vector_alloc(this->numCoef);
	gsl_vector* gradientM = gsl_vector_alloc(this->numCoef);
	gsl_vector* gradientY = gsl_vector_alloc(this->numCoef); 
	gsl_matrix* hessianM = gsl_matrix_alloc(this->numCoef, this->numCoef);
	gsl_matrix* hessianY = gsl_matrix_alloc(this->numCoef, this->numCoef);

	for (int i = 0; i < this->numCoef; i++){
		gsl_vector_set(M, i, vectorM[i]);
		gsl_vector_set(Y, i, vectorY[i]);
		gsl_vector_set(gradientM, i, gradientVectorM[i]);
		gsl_vector_set(gradientY, i, gradientVectorY[i]);
		for (int j = 0; j < this->numCoef; j++){
			gsl_matrix_set(hessianM, i, j, hessianMatrixM[i*this->numCoef+j]);
			gsl_matrix_set(hessianY, i, j, hessianMatrixY[i*this->numCoef+j]);
		}
	}

	gsl_matrix_memcpy(workSpaceMatrix, hessianM);
	gsl_linalg_LU_decomp(hessianM, p, &s);
	detHM = gsl_linalg_LU_det(hessianM, s);
	gsl_linalg_LU_solve(hessianM, p, gradientM, workSpaceVector);
	gsl_vector_sub(workSpaceVector, M);
	gsl_vector_add(workSpaceVector, Y);
	gsl_blas_dgemv(CblasNoTrans, 1, workSpaceMatrix, workSpaceVector, 0, gradientM);
	gsl_blas_ddot(workSpaceVector, gradientM, &workSpaceScale);
	piM = -0.5*workSpaceScale*log(sqrt(detHM));

	gsl_matrix_memcpy(workSpaceMatrix, hessianY);
	gsl_linalg_LU_decomp(hessianY, p, &s);
	detHY = gsl_linalg_LU_det(hessianY, s);
	gsl_linalg_LU_solve(hessianY, p, gradientY, workSpaceVector);
	gsl_vector_sub(workSpaceVector, Y);
	gsl_vector_add(workSpaceVector, M);
	gsl_blas_dgemv(CblasNoTrans, 1, workSpaceMatrix, workSpaceVector, 0, gradientY);
	gsl_blas_ddot(workSpaceVector, gradientY, &workSpaceScale);
	piY = -0.5*workSpaceScale*log(sqrt(detHY));

	alpha = min(0.0, lnLikelihoodt1 - lnLikelihoodt0 + piM - piY);
	return alpha;
}

template <typename chainType>
void stochasticNewtonMCMCChain<chainType>::startPoint(double QoI[]){

	for (int i = 0; i < this->numCoef; i++){
		this->sampleCurrent[i] = QoI[i];
		this->sampleProposal[i] = QoI[i];
	}

	this->samplerSolver->updateParameter(this->sampleProposal.get(), this->numCoef);
	this->samplerSolver->adSolver(this->sampleProposal.get(), gradientVectorM.get(), hessianMatrixM.get(), this->numCoef);
	this->lnLikelihoodt0 = this->samplerSolver->lnLikelihood(this->sampleCurrent.get(), this->numCoef);

	SPDRegularization(hessianMatrixM.get(), this->numCoef, this->samplerSolver->cut_off);

	std::cout << "start point likelihoodt0: " << this->lnLikelihoodt0 << std::endl << std::endl;
}

template <typename chainType>
void stochasticNewtonMCMCChain<chainType>::runStep(){
	sampleGenerator->updatePreMatrix(hessianMatrixM.get());
	sampleGenerator->RWProposalGenerate(this->sampleProposal.get());
	newtonStep(gradientVectorM.get(), hessianMatrixM.get(), this->numCoef, newtonStepArray.get());

	for (int i = 0 ; i < this->numCoef; i++){
		this->sampleProposal[i] = this->sampleCurrent[i] - newtonStepArray[i] + RWSwitch*this->sampleProposal[i];
		// std::cout << newtonStepArray[i] << " " << this->sampleProposal[i] << " ";
	}
	// std::cout << std::endl;

	this->samplerSolver->updateParameter(this->sampleProposal.get(), this->numCoef);
	this->samplerSolver->adSolver(this->sampleProposal.get(), gradientVectorY.get(), hessianMatrixY.get(), this->numCoef);
	this->lnLikelihoodt1 = this->samplerSolver->lnLikelihood(this->sampleProposal.get(), this->numCoef);

	SPDRegularization(hessianMatrixY.get(), this->numCoef, this->samplerSolver->cut_off);
	this->alpha = alphaUpdate(this->lnLikelihoodt0, this->lnLikelihoodt1, this->sampleCurrent.get(), this->sampleProposal.get(), gradientVectorM.get(), gradientVectorY.get(), hessianMatrixM.get(), hessianMatrixY.get());

	// Acceptance Check
	this->alpha = min(0.0, this->lnLikelihoodt1-this->lnLikelihoodt0);
	std::cout << "lnLikelihoodt1: " << this->lnLikelihoodt1 << std::endl; 

	if (this->alpha >= log(gsl_rng_uniform(r)) && !isnan(this->lnLikelihoodt1)){
		// Update likelihood
		std::cout << "sample accepted" << std::endl;
		this->samplerSolver->updateInitialState();
		this->MCMCChainIO->chainWriteCoef(this->sampleProposal.get());
		this->accepted = 1;

		// Backup current status
		this->lnLikelihoodt0 = this->lnLikelihoodt1;
		memcpy(this->sampleCurrent.get(), this->sampleProposal.get(), sizeof(double)*this->numCoef);
		memcpy(gradientVectorM.get(), gradientVectorY.get(), sizeof(double)*this->numCoef);
		memcpy(hessianMatrixM.get(), hessianMatrixY.get(), sizeof(double)*this->numCoef*this->numCoef);

	} else {
		std::cout << "sample rejected" << std::endl;
		this->MCMCChainIO->chainWriteCoef(this->sampleCurrent.get());
		this->accepted = 0;
	}
	this->chainLength = this->chainLength + 1;
	std::cout << "Current lnLikelihoodt0: " << this->lnLikelihoodt0 << std::endl; 
}


template <typename chainType>
void stochasticNewtonMCMCChain<chainType>::run(double QoI[]){
	std::cout << "after burning" << std::endl;

	for (int sampleNum = 0; sampleNum < this->maxChainLength; sampleNum++){	
		runStep();
	}
}


