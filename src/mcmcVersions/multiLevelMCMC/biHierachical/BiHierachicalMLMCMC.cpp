#include "BiHierachicalMLMCMC.h"

BiMLMCMC::BiMLMCMC(string outputPath_, int numLevel_, int numCoef_, double reTau_) : outputPath(outputPath_), numLevel(numLevel_), numCoef(numCoef_), reTau(reTau_) {
	generator.seed(randomSeed);

	// gsl_rng_env_setup();
	// T = gsl_rng_default;
	// r = gsl_rng_alloc(T);

	// gsl_rng_set(r, randomSeed);


};

double BiMLMCMC::runOneChain(string outputPath,int numLevel, int sampleL, int solverL, int chainID, double reTau, int numCoef, int procid, bool validation = 1){
	uniform_real_distribution<double> acceptance(0, 1);

	int I;
	int M;

	if (sampleL == 0 && solverL == 0){
		M = floor(pow(2, 2*numLevel))+1;
	} else if (sampleL!=0 && solverL!=0){
		M = pow(sampleL+solverL, 2)*pow(2, 2*(numLevel-sampleL-solverL));
	} else {
		M = min(pow(2, 2*(numLevel-sampleL)), (pow(2, 2*(numLevel-solverL))));
	}	
	M = M*5+10;

	std::shared_ptr<MLMCMCChain> MCMCChain1;
	std::shared_ptr<MLMCMCChain> MCMCChain2;	

	//Initialize MCMC Chains
	MCMCChain1 = std::make_shared<MLMCMCChain>(outputPath, M, chainID, numCoef, reTau, sampleL, solverL, 1000*acceptance(generator)+19*procid);
	if (validation == 1){
		MCMCChain1->turbProp.validationBoundary();
		MCMCChain1->turbProp.validationPressure();
	}
	
	if (sampleL > 0) {
		MCMCChain2 = std::make_shared<MLMCMCChain>(outputPath, M, chainID+1000, numCoef, reTau, sampleL-1, solverL, 1000*acceptance(generator)+31*procid);
		if (validation == 1){
			MCMCChain2->turbProp.validationBoundary();
			MCMCChain2->turbProp.validationPressure();
		}
	}

	//update beta values
	MCMCChain1->turbProp.updateBeta1(MCMCChain1->approximateCoefProposal, 1, 1);
	if (sampleL > 0){
		MCMCChain2->turbProp.updateBeta1(MCMCChain1->approximateCoefProposal, 1, 1);
	}

	MCMCChain1->solve();
	MCMCChain1->updateLikelihood();
	memcpy(MCMCChain1->xVelocity, MCMCChain1->turbProp.xVelocity, MCMCChain1->turbProp.m*sizeof(double));
	memcpy(MCMCChain1->beta, MCMCChain1->turbProp.betaML, MCMCChain1->turbProp.m*sizeof(double));

	if (sampleL > 0){
		MCMCChain2->solve();
		MCMCChain2->updateLikelihood();
		memcpy(MCMCChain2->xVelocity, MCMCChain2->turbProp.xVelocity, MCMCChain2->turbProp.m*sizeof(double));
		memcpy(MCMCChain2->beta, MCMCChain2->turbProp.betaML, MCMCChain2->turbProp.m*sizeof(double));
	}
    	
	if (solverL == 0){
		if (sampleL == 0){
			MCMCChain1->Q0 = MCMCChain1->Qupper;
			MCMCChain1->Qmean = MCMCChain1->Q0;
		} else {
			MCMCChain1->Q0 = MCMCChain1->Qupper;
			MCMCChain1->Qmean = MCMCChain1->Q0;
			MCMCChain1->Ap[1] = MCMCChain1->A[1];
			MCMCChain1->Amean[1] = MCMCChain1->Ap[1];
			MCMCChain2->Ap[2] = MCMCChain2->A[2];
			MCMCChain2->Amean[2] = MCMCChain2->Ap[2];
			MCMCChain1->Ap[3] = MCMCChain1->A[3];
			MCMCChain1->Amean[3] = MCMCChain1->Ap[3];
			MCMCChain2->Ap[4] = MCMCChain2->A[4];
			MCMCChain2->Amean[4] = MCMCChain2->Ap[4];
			MCMCChain2->Ap[5] = MCMCChain2->A[5];
			MCMCChain2->Amean[5] = MCMCChain2->Ap[5];
			MCMCChain1->Ap[6] = MCMCChain1->A[6];
			MCMCChain1->Amean[6] = MCMCChain1->Ap[6];
			MCMCChain1->Ap[7] = MCMCChain1->A[7];
			MCMCChain1->Amean[7] = MCMCChain1->Ap[7];
			MCMCChain2->Ap[8] = MCMCChain2->A[8];
			MCMCChain2->Amean[8] = MCMCChain2->Ap[8];
			MCMCChain2->Q0 = MCMCChain2->Qupper;
			MCMCChain2->Qmean = MCMCChain2->Q0;
		}
	} else {
		if (sampleL == 0){
			MCMCChain1->Q0 = MCMCChain1->Qupper - MCMCChain1->Qlower;
			MCMCChain1->Qmean = MCMCChain1->Q0;
		} else {
			MCMCChain1->Q0 = MCMCChain1->Qupper - MCMCChain1->Qlower;
			MCMCChain2->Q0 = MCMCChain2->Qupper - MCMCChain2->Qlower;
			MCMCChain1->Qmean = MCMCChain1->Q0;
			MCMCChain2->Qmean = MCMCChain2->Q0;
			MCMCChain1->Ap[1] = MCMCChain1->A[1];
			MCMCChain1->Amean[1] = MCMCChain1->Ap[1];
			MCMCChain2->Ap[2] = MCMCChain2->A[2];
			MCMCChain2->Amean[2] = MCMCChain2->Ap[2];
			MCMCChain1->Ap[3] = MCMCChain1->A[3];
			MCMCChain1->Amean[3] = MCMCChain1->Ap[3];
			MCMCChain2->Ap[4] = MCMCChain2->A[4];
			MCMCChain2->Amean[4] = MCMCChain2->Ap[4];
			MCMCChain2->Ap[5] = MCMCChain2->A[5];
			MCMCChain2->Amean[5] = MCMCChain2->Ap[5];
			MCMCChain1->Ap[6] = MCMCChain1->A[6];
			MCMCChain1->Amean[6] = MCMCChain1->Ap[6];
			MCMCChain1->Ap[7] = MCMCChain1->A[7];
			MCMCChain1->Amean[7] = MCMCChain1->Ap[7];
			MCMCChain2->Ap[8] = MCMCChain2->A[8];
			MCMCChain2->Amean[8] = MCMCChain2->Ap[8];
		}
	}
	double processOutput = 0;

	///////////////////////////////////////Chain Start////////////////////////////////////////////
	while (MCMCChain1->chainLength < MCMCChain1->MLl){
		//forward solve
		MCMCChain1->run();
		//cout << endl << "chainID: " << chainID << " lnLikelihoodt1: " << MCMCChain1->lnLikelihoodt1 << " lnLikelihoodt0: " << MCMCChain1->lnLikelihoodt0 << endl;
		if (sampleL > 0){
			MCMCChain1->lnLikelihoodref = MCMCChain2->returnLikelihoodref(MCMCChain1->approximateCoefProposal);
			MCMCChain2->run();
			MCMCChain2->lnLikelihoodref = MCMCChain1->returnLikelihoodref(MCMCChain2->approximateCoefProposal);
            
			if (solverL == 0){
				MCMCChain1->Q = MCMCChain1->turbPropUpper.velocityInt();
				MCMCChain2->Q = MCMCChain2->turbPropUpper.velocityInt();
			} else {
				MCMCChain1->Q = MCMCChain1->turbPropUpper.velocityInt() - MCMCChain1->turbPropLower.velocityInt();
				MCMCChain2->Q = MCMCChain2->turbPropUpper.velocityInt() - MCMCChain2->turbPropLower.velocityInt();
			}
            	
			if (MCMCChain1->lnLikelihoodt1 - MCMCChain1->lnLikelihoodref > 0){
				I = 1;
			} else {
				I = 0;
			}
			MCMCChain1->A[1] = (1-exp(-MCMCChain1->lnLikelihoodt1+MCMCChain1->lnLikelihoodref))*(MCMCChain1->Q)*I;
			MCMCChain1->A[3] = (exp(-MCMCChain1->lnLikelihoodt1+MCMCChain1->lnLikelihoodref)-1)*I;
			MCMCChain1->A[6] = exp(-MCMCChain1->lnLikelihoodt1+MCMCChain1->lnLikelihoodref)*(MCMCChain1->Q)*I;
			MCMCChain1->A[7] = (MCMCChain1->Q)*(1-I);
                    	
                    	
			if (MCMCChain2->lnLikelihoodref -MCMCChain2->lnLikelihoodt1 > 0){
				I = 1;
			} else {
				I = 0;
			}
			MCMCChain2->A[2] = (exp(-MCMCChain2->lnLikelihoodt1+MCMCChain2->lnLikelihoodref)-1)*(MCMCChain2->Q)*(1-I);
			MCMCChain2->A[4] = (MCMCChain2->Q)*I;
			MCMCChain2->A[5] = (1-exp(-MCMCChain2->lnLikelihoodt1+MCMCChain2->lnLikelihoodref))*(1-I);
			MCMCChain2->A[8] = exp(-MCMCChain2->lnLikelihoodt1+MCMCChain2->lnLikelihoodref)*(MCMCChain2->Q)*(1-I);
                    	

			// Acceptance Check		
			if (MCMCChain1->alpha >= log(acceptance(generator))){
				// Update likelihoodcout
				MCMCChain1->lnLikelihoodt0 = MCMCChain1->lnLikelihoodt1;
				MCMCChain1->Q0             = MCMCChain1->Q;
				memcpy(MCMCChain1->approximateCoef, MCMCChain1->approximateCoefProposal, numCoef*sizeof(double));
				memcpy(MCMCChain1->xVelocity, MCMCChain1->turbProp.xVelocity, MCMCChain1->turbProp.m*sizeof(double));
				memcpy(MCMCChain1->beta, MCMCChain1->turbProp.betaML, MCMCChain1->turbProp.m*sizeof(double));
                    	
				// IO of MCMC Chains
				MCMCChain1->MCMCChainIO->chainWriteQ(MCMCChain1->Q);
				MCMCChain1->MCMCChainIO->chainWriteVelocity(MCMCChain1->turbProp.xVelocity, MCMCChain1->turbProp.m);
				MCMCChain1->MCMCChainIO->chainWriteBeta(MCMCChain1->turbProp.betaML, MCMCChain1->turbProp.m);
				MCMCChain1->MCMCChainIO->chainWriteCoef(MCMCChain1->approximateCoefProposal);
				MCMCChain1->MCMCChainIO->chainWriteLikelihood(MCMCChain1->lnLikelihoodt0);
				MCMCChain1->MCMCChainIO->chainWriteA1(MCMCChain1->A[1]);
				MCMCChain1->MCMCChainIO->chainWriteA3(MCMCChain1->A[3]);
				MCMCChain1->MCMCChainIO->chainWriteA6(MCMCChain1->A[6]);
				MCMCChain1->MCMCChainIO->chainWriteA7(MCMCChain1->A[7]);

                    	
				// Update mean value
				MCMCChain1->Amean[1] = MCMCChain1->Amean[1]*(MCMCChain1->chainLength-1)/MCMCChain1->chainLength + MCMCChain1->A[1]/MCMCChain1->chainLength;
				MCMCChain1->Amean[3] = MCMCChain1->Amean[3]*(MCMCChain1->chainLength-1)/MCMCChain1->chainLength + MCMCChain1->A[3]/MCMCChain1->chainLength;
				MCMCChain1->Amean[6] = MCMCChain1->Amean[6]*(MCMCChain1->chainLength-1)/MCMCChain1->chainLength + MCMCChain1->A[6]/MCMCChain1->chainLength;
				MCMCChain1->Amean[7] = MCMCChain1->Amean[7]*(MCMCChain1->chainLength-1)/MCMCChain1->chainLength + MCMCChain1->A[7]/MCMCChain1->chainLength;
				
				
				// Backup current status
				MCMCChain1->Ap[1] = MCMCChain1->A[1];
				MCMCChain1->Ap[3] = MCMCChain1->A[3];			
				MCMCChain1->Ap[6] = MCMCChain1->A[6];
				MCMCChain1->Ap[7] = MCMCChain1->A[7];
			} else {
				//IO of MCMC Chains
				MCMCChain1->MCMCChainIO->chainWriteA1(MCMCChain1->Ap[1]);
				MCMCChain1->MCMCChainIO->chainWriteA3(MCMCChain1->Ap[3]);
				MCMCChain1->MCMCChainIO->chainWriteA6(MCMCChain1->Ap[6]);
				MCMCChain1->MCMCChainIO->chainWriteA7(MCMCChain1->Ap[7]);
		        MCMCChain1->MCMCChainIO->chainWriteCoef(MCMCChain1->approximateCoef);                       
				MCMCChain1->MCMCChainIO->chainWriteVelocity(MCMCChain1->xVelocity, MCMCChain1->turbProp.m);              
				MCMCChain1->MCMCChainIO->chainWriteBeta(MCMCChain1->beta, MCMCChain1->turbProp.m);                     
		        MCMCChain1->MCMCChainIO->chainWriteLikelihood(MCMCChain1->lnLikelihoodt0);                   
		        MCMCChain1->MCMCChainIO->chainWriteQ(MCMCChain1->Q0);           
				
				// Update mean value
				MCMCChain1->Amean[1] = MCMCChain1->Amean[1]*(MCMCChain1->chainLength-1)/MCMCChain1->chainLength + MCMCChain1->Ap[1]/MCMCChain1->chainLength;
				MCMCChain1->Amean[3] = MCMCChain1->Amean[3]*(MCMCChain1->chainLength-1)/MCMCChain1->chainLength + MCMCChain1->Ap[3]/MCMCChain1->chainLength;
				MCMCChain1->Amean[6] = MCMCChain1->Amean[6]*(MCMCChain1->chainLength-1)/MCMCChain1->chainLength + MCMCChain1->Ap[6]/MCMCChain1->chainLength;
				MCMCChain1->Amean[7] = MCMCChain1->Amean[7]*(MCMCChain1->chainLength-1)/MCMCChain1->chainLength + MCMCChain1->Ap[7]/MCMCChain1->chainLength;

			}

			if (MCMCChain2->alpha >= log(acceptance(generator))){
				// Update likelihood
				MCMCChain2->lnLikelihoodt0  = MCMCChain2->lnLikelihoodt1;
				MCMCChain2->Q0              = MCMCChain2->Q;
				memcpy(MCMCChain2->approximateCoef, MCMCChain2->approximateCoefProposal, numCoef*sizeof(double));
				memcpy(MCMCChain2->xVelocity, MCMCChain2->turbProp.xVelocity, MCMCChain2->turbProp.m*sizeof(double));
				memcpy(MCMCChain2->beta, MCMCChain2->turbProp.betaML, MCMCChain2->turbProp.m*sizeof(double));

				// IO of MCMC Chain
				MCMCChain2->MCMCChainIO->chainWriteVelocity(MCMCChain2->turbProp.xVelocity, MCMCChain2->turbProp.m);
				MCMCChain2->MCMCChainIO->chainWriteBeta(MCMCChain2->turbProp.betaML, MCMCChain2->turbProp.m);
				MCMCChain2->MCMCChainIO->chainWriteCoef(MCMCChain2->approximateCoefProposal);
				MCMCChain2->MCMCChainIO->chainWriteLikelihood(MCMCChain2->lnLikelihoodt0);
				MCMCChain2->MCMCChainIO->chainWriteA2(MCMCChain2->A[2]);
				MCMCChain2->MCMCChainIO->chainWriteA4(MCMCChain2->A[4]);
				MCMCChain2->MCMCChainIO->chainWriteA5(MCMCChain2->A[5]);
				MCMCChain2->MCMCChainIO->chainWriteA8(MCMCChain2->A[8]);

				MCMCChain2->Amean[2] = MCMCChain2->Amean[2]*(MCMCChain2->chainLength-1)/MCMCChain2->chainLength + MCMCChain2->A[2]/MCMCChain2->chainLength;
				MCMCChain2->Amean[4] = MCMCChain2->Amean[4]*(MCMCChain2->chainLength-1)/MCMCChain2->chainLength + MCMCChain2->A[4]/MCMCChain2->chainLength;
				MCMCChain2->Amean[5] = MCMCChain2->Amean[5]*(MCMCChain2->chainLength-1)/MCMCChain2->chainLength + MCMCChain2->A[5]/MCMCChain2->chainLength;
				MCMCChain2->Amean[8] = MCMCChain2->Amean[8]*(MCMCChain2->chainLength-1)/MCMCChain2->chainLength + MCMCChain2->A[8]/MCMCChain2->chainLength;


				// Backup current status
				MCMCChain2->Ap[2] = MCMCChain2->A[2];
				MCMCChain2->Ap[4] = MCMCChain2->A[4];			
				MCMCChain2->Ap[5] = MCMCChain2->A[5];
				MCMCChain2->Ap[8] = MCMCChain2->A[8];

			} else {
				//IO of MCMC
				MCMCChain2->MCMCChainIO->chainWriteA2(MCMCChain2->Ap[2]);
				MCMCChain2->MCMCChainIO->chainWriteA4(MCMCChain2->Ap[4]);
				MCMCChain2->MCMCChainIO->chainWriteA5(MCMCChain2->Ap[5]);
				MCMCChain2->MCMCChainIO->chainWriteA8(MCMCChain2->Ap[8]);
		        MCMCChain2->MCMCChainIO->chainWriteCoef(MCMCChain2->approximateCoef);						
		        MCMCChain2->MCMCChainIO->chainWriteBeta(MCMCChain2->beta, MCMCChain2->turbProp.m);          
		        MCMCChain2->MCMCChainIO->chainWriteVelocity(MCMCChain2-> xVelocity, MCMCChain2->turbProp.m);    
		        MCMCChain2->MCMCChainIO->chainWriteLikelihood(MCMCChain2->lnLikelihoodt0);

				MCMCChain2->Amean[2] = MCMCChain2->Amean[2]*(MCMCChain2->chainLength-1)/MCMCChain2->chainLength + MCMCChain2->Ap[2]/MCMCChain2->chainLength;
		 		MCMCChain2->Amean[4] = MCMCChain2->Amean[4]*(MCMCChain2->chainLength-1)/MCMCChain2->chainLength + MCMCChain2->Ap[4]/MCMCChain2->chainLength;
				MCMCChain2->Amean[5] = MCMCChain2->Amean[5]*(MCMCChain2->chainLength-1)/MCMCChain2->chainLength + MCMCChain2->Ap[5]/MCMCChain2->chainLength;
				MCMCChain2->Amean[8] = MCMCChain2->Amean[8]*(MCMCChain2->chainLength-1)/MCMCChain2->chainLength + MCMCChain2->Ap[8]/MCMCChain2->chainLength;	
			}
            	
		} else if (sampleL == 0 && solverL == 0) {
			MCMCChain1->Q = MCMCChain1->turbProp.velocityInt();
            	
			if (MCMCChain1->alpha >= log(acceptance(generator))){
				// Update likelihood

				MCMCChain1->lnLikelihoodt0  = MCMCChain1->lnLikelihoodt1;
				MCMCChain1->Q0              = MCMCChain1->Q;
				memcpy(MCMCChain1->approximateCoef, MCMCChain1->approximateCoefProposal, numCoef*sizeof(double));
				memcpy(MCMCChain1->xVelocity, MCMCChain1->turbProp.xVelocity, MCMCChain1->turbProp.m*sizeof(double));
				memcpy(MCMCChain1->beta, MCMCChain1->turbProp.betaML, MCMCChain1->turbProp.m*sizeof(double));
    	
				// IO of MCMC Chain
				MCMCChain1->MCMCChainIO->chainWriteVelocity(MCMCChain1->xVelocity, MCMCChain1->turbProp.m);
				MCMCChain1->MCMCChainIO->chainWriteBeta(MCMCChain1->beta, MCMCChain1->turbProp.m);
				MCMCChain1->MCMCChainIO->chainWriteCoef(MCMCChain1->approximateCoefProposal);
				MCMCChain1->MCMCChainIO->chainWriteQ(MCMCChain1->Q);
				//
				MCMCChain1->Qmean = MCMCChain1->Qmean*(MCMCChain1->chainLength-1)/MCMCChain1->chainLength + MCMCChain1->Q/MCMCChain1->chainLength;
			} else {
				// IO of MCMC Chain
				MCMCChain1->MCMCChainIO->chainWriteVelocity(MCMCChain1->xVelocity, MCMCChain1->turbProp.m);
				MCMCChain1->MCMCChainIO->chainWriteBeta(MCMCChain1->beta, MCMCChain1->turbProp.m);
			    MCMCChain1->MCMCChainIO->chainWriteCoef(MCMCChain1->approximateCoef);
				MCMCChain1->MCMCChainIO->chainWriteQ(MCMCChain1->Q0);
				//
				MCMCChain1->Qmean = MCMCChain1->Qmean*(MCMCChain1->chainLength-1)/MCMCChain1->chainLength + MCMCChain1->Q0/MCMCChain1->chainLength;
			}
		} else {
			MCMCChain1->Q = MCMCChain1->turbPropUpper.velocityInt() - MCMCChain1->turbPropLower.velocityInt();
			if (MCMCChain1->alpha >= log(acceptance(generator))){
            	
				MCMCChain1->lnLikelihoodt0  = MCMCChain1->lnLikelihoodt1;
				MCMCChain1->Q0              = MCMCChain1->Q;
				memcpy(MCMCChain1->approximateCoef, MCMCChain1->approximateCoefProposal, numCoef*sizeof(double));
				memcpy(MCMCChain1->xVelocity, MCMCChain1->turbProp.xVelocity, MCMCChain1->turbProp.m*sizeof(double));
				memcpy(MCMCChain1->beta, MCMCChain1->turbProp.betaML, MCMCChain1->turbProp.m*sizeof(double));


				// IO of MCMC Chain
				MCMCChain1->MCMCChainIO->chainWriteQ(MCMCChain1->Q);
				MCMCChain1->MCMCChainIO->chainWriteVelocity(MCMCChain1->xVelocity, MCMCChain1->turbProp.m);
				MCMCChain1->MCMCChainIO->chainWriteBeta(MCMCChain1->beta, MCMCChain1->turbProp.m);
				MCMCChain1->MCMCChainIO->chainWriteCoef(MCMCChain1->approximateCoefProposal);
            	
				// update Q mean
				MCMCChain1->Qmean = MCMCChain1->Qmean*(MCMCChain1->chainLength-1)/MCMCChain1->chainLength + MCMCChain1->Q0/MCMCChain1->chainLength;

			} else {
				// IO of MCMC Chain
				MCMCChain1->MCMCChainIO->chainWriteVelocity(MCMCChain1->xVelocity, MCMCChain1->turbProp.m);
				MCMCChain1->MCMCChainIO->chainWriteBeta(MCMCChain1->beta, MCMCChain1->turbProp.m);
			    MCMCChain1->MCMCChainIO->chainWriteCoef(MCMCChain1->approximateCoef);
				MCMCChain1->MCMCChainIO->chainWriteQ(MCMCChain1->Q0);		        
			
				// update Q Mean
				MCMCChain1->Qmean = MCMCChain1->Qmean*(MCMCChain1->chainLength-1)/MCMCChain1->chainLength + MCMCChain1->Q0/MCMCChain1->chainLength;
			}	
		}
	}
	if (sampleL == 0){
		processOutput = MCMCChain1->Qmean;
	} else {
		processOutput = MCMCChain1->Amean[1] + MCMCChain2->Amean[2] + MCMCChain1->Amean[3]*(MCMCChain2->Amean[4]+MCMCChain2->Amean[8]) + MCMCChain2->Amean[5]*(MCMCChain1->Amean[6]+MCMCChain1->Amean[7]);
	}
	cout << chainID << " expectation: " << processOutput << " acceptance rate: " << MCMCChain1->coefProposal->acceptanceRate << endl;
	return processOutput;
}


double BiMLMCMC::run(int procid, int numprocs){
	int chainID = 0;
	finalSum = 0;
	if (numprocs > 0){}
	for (int i = 0; i < numLevel; i++){
		for (int j = 0; j < numLevel-i; j++){
			//if (procid == chainID%(2*numprocs) || procid == 2*numprocs-chainID%(2*numprocs)-1){
			finalExpectation = runOneChain(outputPath, numLevel, i, j, chainID, 550, numCoef, procid, 1);
			finalSum = finalSum + finalExpectation;
			//}
			chainID = chainID + 1;
		}
	}
	return finalSum;
}
