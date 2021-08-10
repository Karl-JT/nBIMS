#include "MCMCChain.h"

template<typename chainType>
class MLMCMCChain{
public:
	int samplerLevel;

	std::unique_ptr<double []> Q;
	std::unique_ptr<double []> Qmean;
	std::array<std::unique_ptr<double []>, 9> A;
	std::array<std::unique_ptr<double []>, 9> Amean;

	//Solver Setup
	std::shared_ptr<chainType> upperSamplerChain;
	std::shared_ptr<chainType> lowerSamplerChain;
	
	double upperSamplerRef;
	double lowerSamplerRef;
	double upperSamplerI;
	double lowerSamplerI;

	MLMCMCChain(string outputPath, int Ml_, int procid_, int numCoef_, int L_);
	MLMCMCChain(string outputPath, int Ml_, int procid_, int numCoef_, int L_, double cut_off);
	~MLMCMCChain(){};

	virtual void startPoint(double QoI[]);
	virtual void solve(){};
	virtual void runStep();
	virtual void burnin(double singleChainOutput[]);
	virtual void run(double singleChainOutput[]);
};


template<typename chainType>
MLMCMCChain<chainType>::MLMCMCChain(string outputPath, int Ml_, int procid_, int numCoef_, int L_) : samplerLevel(L_) {
	upperSamplerChain = make_shared<chainType>(outputPath, Ml_, procid_, numCoef_, L_);
	if (samplerLevel > 0){
		lowerSamplerChain = make_shared<chainType>(outputPath, Ml_, procid_, numCoef_, L_-1);
	}
	Q = std::make_unique<double []>(numCoef_);
	Qmean = std::make_unique<double []>(numCoef_);
	for (int i = 1; i < 9; i++){
		A[i] = std::make_unique<double []>(numCoef_);
		Amean[i] = std::make_unique<double []>(numCoef_);
	}
}

template<typename chainType>
MLMCMCChain<chainType>::MLMCMCChain(string outputPath, int Ml_, int procid_, int numCoef_, int L_, double cut_off_) : samplerLevel(L_) {
	upperSamplerChain = make_shared<chainType>(outputPath, Ml_, procid_, numCoef_, L_, cut_off_);
	if (samplerLevel > 0){
		lowerSamplerChain = make_shared<chainType>(outputPath, Ml_, procid_, numCoef_, L_-1, cut_off_);
	}
	Q = std::make_unique<double []>(numCoef_);
	Qmean = std::make_unique<double []>(numCoef_);
	for (int i = 1; i < 9; i++){
		A[i] = std::make_unique<double []>(numCoef_);
		Amean[i] = std::make_unique<double []>(numCoef_);
	}
}

template<typename chainType>
void MLMCMCChain<chainType>::startPoint(double QoI[]){
	// upperSamplerChain->startPoint(upperSamplerChain->sampleCurrent.get());
	// if (samplerLevel > 0){
	// 	lowerSamplerChain->startPoint(lowerSamplerChain->sampleCurrent.get());
	// }
	upperSamplerChain->startPoint(QoI);
	if (samplerLevel > 0){
		lowerSamplerChain->startPoint(QoI);
	}
}

template<typename chainType>
void MLMCMCChain<chainType>::runStep(){
	upperSamplerChain->runStep();
	if (samplerLevel > 0){
		upperSamplerRef = lowerSamplerChain->returnLikelihoodref(upperSamplerChain->sampleProposal.get());
		lowerSamplerChain->runStep();
		lowerSamplerRef = upperSamplerChain->returnLikelihoodref(lowerSamplerChain->sampleProposal.get());
	}

	if (upperSamplerChain->accepted){
		// Update likelihood
		if (samplerLevel > 0){
			if (-upperSamplerChain->lnLikelihoodt1+upperSamplerRef <= 0){
				upperSamplerI = 1;
			} else {
				upperSamplerI = 0;
			}
			for (int i = 0; i < upperSamplerChain->numCoef; i++){
				A[1][i] = (1-exp(-upperSamplerChain->lnLikelihoodt1+upperSamplerRef))*upperSamplerI*upperSamplerChain->sampleProposal[i];
				A[3][i] = (exp(-upperSamplerChain->lnLikelihoodt1+upperSamplerRef)-1)*upperSamplerI;
				A[6][i] = exp(-upperSamplerChain->lnLikelihoodt1+upperSamplerRef)*upperSamplerI*upperSamplerChain->sampleProposal[i];
				A[7][i] = (1-upperSamplerI)*upperSamplerChain->sampleProposal[i];
	
				Amean[1][i] = A[1][i]/(upperSamplerChain->chainLength+1) + Amean[1][i]*(upperSamplerChain->chainLength)/(upperSamplerChain->chainLength+1);
				Amean[3][i] = A[3][i]/(upperSamplerChain->chainLength+1) + Amean[3][i]*(upperSamplerChain->chainLength)/(upperSamplerChain->chainLength+1);
				Amean[6][i] = A[6][i]/(upperSamplerChain->chainLength+1) + Amean[6][i]*(upperSamplerChain->chainLength)/(upperSamplerChain->chainLength+1);
				Amean[7][i] = A[7][i]/(upperSamplerChain->chainLength+1) + Amean[7][i]*(upperSamplerChain->chainLength)/(upperSamplerChain->chainLength+1);

				// upperSamplerChain->MCMCChainIO->chainWriteA1(upperSamplerChain->A[1]);
				// upperSamplerChain->MCMCChainIO->chainWriteA3(upperSamplerChain->A[3]);
				// upperSamplerChain->MCMCChainIO->chainWriteA6(upperSamplerChain->A[6]);
				// upperSamplerChain->MCMCChainIO->chainWriteA7(upperSamplerChain->A[7]); 
			}
		} else {
			for (int i = 0; i < upperSamplerChain->numCoef; i++){
				Q[i]     = upperSamplerChain->sampleProposal[i];
				Qmean[i] = Q[i]/(upperSamplerChain->chainLength+1) + Qmean[i]*(upperSamplerChain->chainLength)/(upperSamplerChain->chainLength+1);
			}
		}

		// upperSamplerChain->MCMCChainIO->chainWriteCoef(upperSamplerChain->sampleProposal);
		// upperSamplerChain->MCMCChainIO->chainWriteLikelihood(upperSamplerChain->lnLikelihoodt0);

		// Backup current status
		for (int i = 0; i < upperSamplerChain->numCoef; i++){
			upperSamplerChain->sampleCurrent[i] = upperSamplerChain->sampleProposal[i];
		}
	} else {
		if (samplerLevel > 0){
			// upperSamplerChain->MCMCChainIO->chainWriteA1(upperSamplerChain->A[1]);
			// upperSamplerChain->MCMCChainIO->chainWriteA3(upperSamplerChain->A[3]);
			// upperSamplerChain->MCMCChainIO->chainWriteA6(upperSamplerChain->A[6]);
			// upperSamplerChain->MCMCChainIO->chainWriteA7(upperSamplerChain->A[7]);	

			for (int i = 0; i < upperSamplerChain->numCoef; i++){
				Amean[1][i] = A[1][i]/(upperSamplerChain->chainLength+1) + Amean[1][i]*(upperSamplerChain->chainLength)/(upperSamplerChain->chainLength+1);
				Amean[3][i] = A[3][i]/(upperSamplerChain->chainLength+1) + Amean[3][i]*(upperSamplerChain->chainLength)/(upperSamplerChain->chainLength+1);
				Amean[6][i] = A[6][i]/(upperSamplerChain->chainLength+1) + Amean[6][i]*(upperSamplerChain->chainLength)/(upperSamplerChain->chainLength+1);
				Amean[7][i] = A[7][i]/(upperSamplerChain->chainLength+1) + Amean[7][i]*(upperSamplerChain->chainLength)/(upperSamplerChain->chainLength+1);
			}
		} else {

			for (int i = 0; i < upperSamplerChain->numCoef; i++){
				Qmean[i] = Q[i]/(upperSamplerChain->chainLength+1) + Qmean[i]*(upperSamplerChain->chainLength)/(upperSamplerChain->chainLength+1);
			}
		}
	    // upperSamplerChain->MCMCChainIO->chainWriteCoef(upperSamplerChain->approximateCoef);
	    // upperSamplerChain->MCMCChainIO->chainWriteLikelihood(upperSamplerChain->lnLikelihoodt0);
	}



	if (samplerLevel > 0){ 
		if (lowerSamplerChain->accepted){
			// Update likelihood
			if (-lowerSamplerRef+lowerSamplerChain->lnLikelihoodt1 <= 0){
				lowerSamplerI = 1;
			} else {
				lowerSamplerI = 0;
			}


			for (int i = 0; i < upperSamplerChain->numCoef; i++){

				if (lowerSamplerI == 1){
					A[2][i] = 0;
					A[4][i] = lowerSamplerI*lowerSamplerChain->sampleProposal[i];
					A[5][i] = 0;
					A[8][i] = 0;
				} else {
					A[2][i] = (exp(-lowerSamplerChain->lnLikelihoodt1+lowerSamplerRef)-1)*lowerSamplerChain->sampleProposal[i];
					A[4][i] = lowerSamplerI*lowerSamplerChain->sampleProposal[i];
					A[5][i] = (1-exp(-lowerSamplerChain->lnLikelihoodt1+lowerSamplerRef));
					A[8][i] = exp(-lowerSamplerChain->lnLikelihoodt1+lowerSamplerRef)*lowerSamplerChain->sampleProposal[i];					
				}

				// A[2][i] = (1-lowerSamplerI)*(exp(-lowerSamplerChain->lnLikelihoodt1+lowerSamplerRef)-1)*lowerSamplerChain->sampleProposal[i];
				// A[4][i] = lowerSamplerI*lowerSamplerChain->sampleProposal[i];
				// A[5][i] = (1-lowerSamplerI)*(1-exp(-lowerSamplerChain->lnLikelihoodt1+lowerSamplerRef));
				// A[8][i] = (1-lowerSamplerI)*exp(-lowerSamplerChain->lnLikelihoodt1+lowerSamplerRef)*lowerSamplerChain->sampleProposal[i];	

				Amean[2][i] = A[2][i]/(lowerSamplerChain->chainLength+1) + Amean[2][i]*(lowerSamplerChain->chainLength)/(lowerSamplerChain->chainLength+1);
				Amean[4][i] = A[4][i]/(lowerSamplerChain->chainLength+1) + Amean[4][i]*(lowerSamplerChain->chainLength)/(lowerSamplerChain->chainLength+1);
				Amean[5][i] = A[5][i]/(lowerSamplerChain->chainLength+1) + Amean[5][i]*(lowerSamplerChain->chainLength)/(lowerSamplerChain->chainLength+1);
				Amean[8][i] = A[8][i]/(lowerSamplerChain->chainLength+1) + Amean[8][i]*(lowerSamplerChain->chainLength)/(lowerSamplerChain->chainLength+1);

				// lowerSamplerChain->MCMCChainIO->chainWriteA2(lowerSamplerChain->A[2]);
				// lowerSamplerChain->MCMCChainIO->chainWriteA4(lowerSamplerChain->A[4]);
				// lowerSamplerChain->MCMCChainIO->chainWriteA5(lowerSamplerChain->A[5]);
				// lowerSamplerChain->MCMCChainIO->chainWriteA8(lowerSamplerChain->A[8]);	
			}

			// lowerSamplerChain->MCMCChainIO->chainWriteCoef(lowerSamplerChain->sampleProposal);
			// lowerSamplerChain->MCMCChainIO->chainWriteLikelihood(lowerSamplerChain->lnLikelihoodt0);

			// Backup current status
			for (int i = 0; i < lowerSamplerChain->numCoef; i++){
				lowerSamplerChain->sampleCurrent[i] = lowerSamplerChain->sampleProposal[i];
			}
		} else {
			// lowerSamplerChain->MCMCChainIO->chainWriteA2(lowerSamplerChain->A[2]);
			// lowerSamplerChain->MCMCChainIO->chainWriteA4(lowerSamplerChain->A[4]);
			// lowerSamplerChain->MCMCChainIO->chainWriteA5(lowerSamplerChain->A[5]);
			// lowerSamplerChain->MCMCChainIO->chainWriteA8(lowerSamplerChain->A[8]);	

			for (int i = 0; i < upperSamplerChain->numCoef; i++){
				Amean[2][i] = A[2][i]/(lowerSamplerChain->chainLength+1) + Amean[2][i]*(lowerSamplerChain->chainLength)/(lowerSamplerChain->chainLength+1);
				Amean[4][i] = A[4][i]/(lowerSamplerChain->chainLength+1) + Amean[4][i]*(lowerSamplerChain->chainLength)/(lowerSamplerChain->chainLength+1);
				Amean[5][i] = A[5][i]/(lowerSamplerChain->chainLength+1) + Amean[5][i]*(lowerSamplerChain->chainLength)/(lowerSamplerChain->chainLength+1);
				Amean[8][i] = A[8][i]/(lowerSamplerChain->chainLength+1) + Amean[8][i]*(lowerSamplerChain->chainLength)/(lowerSamplerChain->chainLength+1);
			}

			// lowerSamplerChain->MCMCChainIO->chainWriteCoef(lowerSamplerChain->sampleCurrent.get());
			// lowerSamplerChain->MCMCChainIO->chainWriteLikelihood(lowerSamplerChain->lnLikelihoodt0);
		}
	}
}

template<typename chainType>
void MLMCMCChain<chainType>::burnin(double singleChainOutput[]){
	startPoint(singleChainOutput);
	upperSamplerChain->RWSwitch = 0;
	if (samplerLevel > 0){
		lowerSamplerChain->RWSwitch = 0;
	}
	for (int i = 0; i < 25; i++){
		runStep();
		upperSamplerChain->samplerSolver->cut_off = max(1.0, (upperSamplerChain->samplerSolver->cut_off)*0.5);
		if (samplerLevel > 0){
			lowerSamplerChain->samplerSolver->cut_off = max(1.0, (lowerSamplerChain->samplerSolver->cut_off)*0.5);
		}
		std::cout << "//////////////burning////////////" << std::endl;
	}

	upperSamplerChain->RWSwitch = 1;
	if (samplerLevel > 0){
		lowerSamplerChain->RWSwitch = 1;
	}
	runStep();
	upperSamplerChain->startPoint(upperSamplerChain->sampleProposal.get());
	if (samplerLevel > 0){
		lowerSamplerChain->startPoint(lowerSamplerChain->sampleProposal.get());
	}

	upperSamplerChain->chainLength = 0;
	if (samplerLevel > 0){
		lowerSamplerChain->chainLength = 0;
	}

	runStep();

	for (int i = 0; i < upperSamplerChain->numCoef; i++){
		Qmean[i] = Q[i];;
		Amean[1][i] = A[1][i];
		Amean[2][i] = A[2][i];
		Amean[3][i] = A[3][i];
		Amean[4][i] = A[4][i];
		Amean[5][i] = A[5][i];
		Amean[6][i] = A[6][i];
		Amean[7][i] = A[7][i];
		Amean[8][i] = A[8][i];
	}
};


template<typename chainType>
void MLMCMCChain<chainType>::run(double singleChainOutput[]){
	burnin(singleChainOutput);
	for (int i = 1; i < upperSamplerChain->maxChainLength; i++){
		// cout << "upperSamplerChain " << upperSamplerChain->chainLength << endl;
		runStep();		
	}
	for (int i = 0; i < upperSamplerChain->numCoef; i++){
		if (samplerLevel > 0){
			singleChainOutput[i] = Amean[1][i] + Amean[2][i] + Amean[3][i]*(Amean[4][i] + Amean[8][i]) + Amean[5][i]*(Amean[6][i]+ Amean[7][i]);
			std::cout << "single chain ouput: " << singleChainOutput[i] << " " << Amean[1][i] << " " << Amean[2][i] << " " << Amean[3][i] << " " << Amean[4][i] << " " << Amean[5][i] << " " << Amean[6][i] << " " << Amean[7][i] << " " << Amean[8][i] << " " << std::endl;
		} else {
			singleChainOutput[i] = Qmean[i];
			std::cout << "single chain output: " << Qmean[i] << std::endl;
		}
	}
}
