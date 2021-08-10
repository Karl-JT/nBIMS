#include "MLMCMCChain.h"


MLMCMCChain::MLMCMCChain(string outputPath, int MLl_, int procid_, int numCoef_, int L_, int l_) : MCMCChain(outputPath, MLl_, procid_, numCoef_, L_), solverLevel(l_) {
	caseInitialize(turbPropUpper, reTau, solverLevel);
	turbPropUpper.validationBoundary();
	turbPropUpper.validationPressure();
	initialization(turbPropUpper);

	if (solverLevel > 0){
		caseInitialize(turbPropLower, reTau, solverLevel-1);
		turbPropLower.validationBoundary();
		turbPropLower.validationPressure();
		initialization(turbPropLower);
	}
	observation = turbProp.validationObservation();
}

void MLMCMCChain::solve(){
	iterativeSolver(turbProp);
	iterativeSolver(turbPropUpper);
	if (solverLevel > 0){
		iterativeSolver(turbPropLower);
	}
}

void MLMCMCChain::run(){
	//Proposal
	coefProposal->compGaussianProposalGenerator(approximateCoefProposal);
	for (int i = 0; i < numCoef; i++){
		approximateCoefProposal[i] = sqrt(1-pow(stepSize, 2))*approximateCoef[i] + stepSize*approximateCoefProposal[i];
	}

	//update beta
	turbProp.updateBeta1(approximateCoefProposal, 1, 1);
	turbPropUpper.updateBeta1(approximateCoefProposal, 1, 1);
	if (solverLevel > 0){
		turbPropLower.updateBeta1(approximateCoefProposal, 1, 1);
	}
	
	// for (int i = 0; i < turbProp.m; i++){
	// 	turbProp.betaML[i] = 0; 
	// 	for (int k = 0; k < numCoef; k++){
	// 		turbProp.betaML[i] = turbProp.betaML[i] + approximateCoefProposal[k]/pow((k+1), decay)*cos(M_PI*(k+1)*turbProp.yCoordinate[i]);
	// 	}
	// 	turbProp.betaML[i] = exp(turbProp.betaML[i]);
	// }

	//solve forward and compute likelihood
	iterativeSolver(turbProp);
	iterativeSolver(turbPropUpper);
	if (solverLevel > 0){
		iterativeSolver(turbPropLower);
	}

	lnLikelihoodt1 = turbProp.validationLnLikelihood(turbProp.validationObservation(), 1);
	// for (int i = 0; i < turbProp.m; i++) {
	// 	lnLikelihoodt1 = lnLikelihoodt1 - 0.5e4*pow(turbProp.xVelocity[i]-validation[i], 2);
	// }
	alpha = min(0.0, lnLikelihoodt1-lnLikelihoodt0);
	coefProposal->updateAcceptanceRate(alpha);
	chainLength++;
}
