#include "../mcmcCore/MCMCBase.h"
#include "../mcmcCore/MCMCMetropolis.h"
#include "../../inverseSolver/solverInterface/solverInterface.h"

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

class plainMCMC : public MCMCBase {
private:
	gsl_rng *r;
	const gsl_rng_type *T;
	chainIO *MCMCChainIO;
public:
	int numCoef = 2;
	int decay =1;

	plainMCMC(string outputPath);
	~plainMCMC(){};

	template<typename problemType>
	void run(problemType& forwardSolver, double QoI[]);

	template<typename problemType>
	void runValidation(problemType& forwardSolver, double QoI[]);
};

plainMCMC::plainMCMC(string outputPath){
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	gsl_rng_set(r, 2019);

	MCMCChainIO = new chainIO(outputPath, 0, numCoef); 
}


template<typename problemType>
void plainMCMC::run(problemType& forwardSolver, double QoI[]){
		//Initialize Proposal Kernal
	default_random_engine generator;
	generator.seed(20191016);
	uniform_real_distribution<double> acceptance(0, 1);

	cout << QoI[0] << endl;

	double stepSize = 0.005; 
	double approximateCoef[numCoef];
	double approximateCoefProposal[numCoef];
	double fourierSeriesCoef[numCoef];

	fourierSeriesCoef[0] = 0.2;
	approximateCoef[0] = 0.2;
	approximateCoefProposal[0] = 0.2;

	fourierSeriesCoef[1] = 0.2;
	approximateCoef[1] = 0.2;
	approximateCoefProposal[1] = 0.2;

	compUniformProposal coefProposal(numCoef, 0, time(NULL));
	coefProposal.scale = 1;

	//Initialize propability variables
	double lnLikelihoodt0 = 0;
	double lnLikelihoodt1 = 0;
	double alpha = 0;

	//update beta values
	forwardSolver.turbProp.updateBeta(fourierSeriesCoef, numCoef, decay);
	
	//solve 
	iterativeSolver(forwardSolver.turbProp);
	lnLikelihoodt0 = forwardSolver.turbProp.lnLikelihood();
	MCMCChainIO->chainWriteCoordinate(forwardSolver.turbProp.yCoordinate, forwardSolver.turbProp.m);
	// for (int i = 0; i < forwardSolver.turbProp.m; i++){
	// 	cout << forwardSolver.turbProp.yCoordinate[i] << ", ";
	// }
	// cout << endl;
	// for (int i = 0; i < forwardSolver.turbProp.m; i++){
	// 	cout << forwardSolver.turbProp.xVelocity[i] << ", ";
	// }
	// cout << endl;
///////////////////////////////////////Chain Start////////////////////////////////////////////
	for (int j = 0; j < 10000; j++){	
		lnLikelihoodt1 = 0;
		coefProposal.compGaussianProposalGenerator(approximateCoefProposal);
		for (int i = 0; i < numCoef; i++){
			approximateCoefProposal[i] = sqrt(1-pow(stepSize, 2))*approximateCoef[i] + stepSize*approximateCoefProposal[i];
			fourierSeriesCoef[i] = approximateCoefProposal[i];//2*gsl_cdf_ugaussian_P(approximateCoefProposal[i])-1;
			cout << fourierSeriesCoef[i] << " ";
		}
		cout << endl;

		// Forward Solving
		forwardSolver.turbProp.updateBeta(fourierSeriesCoef, numCoef, decay);
		iterativeSolver(forwardSolver.turbProp);
		lnLikelihoodt1 = forwardSolver.turbProp.lnLikelihood();

		// Acceptance Check
		alpha = min(0.0, lnLikelihoodt1-lnLikelihoodt0);
		cout << endl << "lnLikelihoodt1: "<< lnLikelihoodt1 << " lnLikelihoodt0: " << lnLikelihoodt0 << " alpha: " << alpha << endl;
		coefProposal.updateAcceptanceRate(alpha);
		cout << "acceptanceRate: " << coefProposal.acceptanceRate << endl;
		if (alpha >= log(acceptance(generator))){
			// Update likelihood
			cout << "sample accepted" << endl;
			lnLikelihoodt0 = lnLikelihoodt1;

			MCMCChainIO->chainWriteBeta(forwardSolver.turbProp.betaML, forwardSolver.turbProp.m);
			MCMCChainIO->chainWriteVelocity(forwardSolver.turbProp.xVelocity, forwardSolver.turbProp.m);
			MCMCChainIO->chainWriteCoef(approximateCoefProposal);

			// Backup current status
			for (int i = 0; i < numCoef; i++){
				approximateCoef[i] = approximateCoefProposal[i];
			}
		} else {
			cout << "sample rejected" << endl;
			for (int i = 0; i < numCoef; i++){
				approximateCoefProposal[i] = approximateCoef[i];
			}
			MCMCChainIO->chainWriteCoef(approximateCoefProposal);
		}
		cout << "Current lnLikelihoodt0: " << lnLikelihoodt0 << endl; 
	}
}


template<typename problemType>
void plainMCMC::runValidation(problemType& forwardSolver, double QoI[]){
	forwardSolver.turbProp.validationBoundary();
	forwardSolver.turbProp.validationPressure();


	cout << forwardSolver.turbProp.boundaryVelocity[0] << " " << forwardSolver.turbProp.boundaryVelocity[1] << endl;
	//Initialize Proposal Kernal
	default_random_engine generator;
	generator.seed(20191016);
	uniform_real_distribution<double> acceptance(0, 1);

	double stepSize = 1; 
	double approximateCoef[numCoef];
	double approximateCoefProposal[numCoef];
	double fourierSeriesCoef[numCoef];

	compUniformProposal coefProposal(numCoef, 0);
	coefProposal.scale = stepSize;

	//Initialize propability variables
	double lnLikelihoodt0 = 0;
	double lnLikelihoodt1 = 0;
	double workspace = forwardSolver.turbProp.velocityInt();
	double alpha = 0;

	//update beta values
	forwardSolver.turbProp.updateBeta1(fourierSeriesCoef, numCoef, decay);
	cout << "reference" << forwardSolver.turbProp.validationObservation() << endl;
	//solve 
	iterativeSolver(forwardSolver.turbProp);
	lnLikelihoodt0 = forwardSolver.turbProp.validationLnLikelihood(forwardSolver.turbProp.validationObservation(), 1);
	MCMCChainIO->chainWriteCoordinate(forwardSolver.turbProp.yCoordinate, forwardSolver.turbProp.m);
	for (int i = 0; i < forwardSolver.turbProp.m; i++){
		cout << forwardSolver.turbProp.yCoordinate[i] << ", ";
	}
	cout << endl;
	for (int i = 0; i < forwardSolver.turbProp.m; i++){
		cout << forwardSolver.turbProp.xVelocity[i] << ", ";
	}
	cout << endl;
///////////////////////////////////////Chain Start////////////////////////////////////////////
	for (int j = 0; j < 1000000; j++){	
		lnLikelihoodt1 = 0;
		coefProposal.compGaussianProposalGenerator(approximateCoefProposal);
		for (int i = 0; i < numCoef; i++){
			approximateCoefProposal[i] = sqrt(1-pow(stepSize, 2))*approximateCoef[i] + stepSize*approximateCoefProposal[i];
			cout << approximateCoefProposal[i] << " ";
			fourierSeriesCoef[i] = approximateCoefProposal[i];//2*gsl_cdf_ugaussian_P(approximateCoefProposal[i])-1;
		}
		cout << endl;

		// Forward Solving
		forwardSolver.turbProp.updateBeta1(fourierSeriesCoef, numCoef, decay);
		iterativeSolver(forwardSolver.turbProp);
		lnLikelihoodt1 = forwardSolver.turbProp.validationLnLikelihood(forwardSolver.turbProp.validationObservation(), 1);

		// Acceptance Check
		alpha = min(0.0, lnLikelihoodt1-lnLikelihoodt0);
		cout << endl << "lnLikelihoodt1: "<< lnLikelihoodt1 << " lnLikelihoodt0: " << lnLikelihoodt0 << " alpha: " << alpha << endl;
		coefProposal.updateAcceptanceRate(alpha);
		cout << "stepSize: " << stepSize << endl;
		if (alpha >= log(acceptance(generator))){
			// Update likelihood
			cout << "sample accepted" << endl;
			lnLikelihoodt0 = lnLikelihoodt1;

			MCMCChainIO->chainWriteBeta(forwardSolver.turbProp.betaML, forwardSolver.turbProp.m);
			MCMCChainIO->chainWriteVelocity(forwardSolver.turbProp.xVelocity, forwardSolver.turbProp.m);
			MCMCChainIO->chainWriteCoef(approximateCoefProposal);
			MCMCChainIO->chainWriteQ(forwardSolver.turbProp.velocityInt());
			workspace = forwardSolver.turbProp.velocityInt();

			// Backup current status
			for (int i = 0; i < numCoef; i++){
				approximateCoef[i] = approximateCoefProposal[i];
			}
		} else {
			cout << "sample rejected" << endl;
			for (int i = 0; i < numCoef; i++){
				approximateCoefProposal[i] = approximateCoef[i];
			}
			MCMCChainIO->chainWriteCoef(approximateCoefProposal);
			MCMCChainIO->chainWriteQ(workspace);
		}
		cout << "Current lnLikelihoodt0: " << lnLikelihoodt0 << endl; 
	}
}

