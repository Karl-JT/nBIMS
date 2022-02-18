#include "MLMCMC_Bi_Uniform.h"

class templateSolver{
private:
public:
	double samples[10];

	templateSolver(int levels, int sampleSize, double noiseVariance){};
	~templateSolver(){};

	virtual void solve(){};
	virtual double lnLikelihood(){};
	virtual void solve4Obs(){};
	virtual double solve4QoI(){};
	virtual double getAlpha(double lnLikelihoodt0, double lnLikelihoodt1){};
	virtual void priorSample(double initialSamples[]){};
};

int main(){
	MLMCMC_Bi_Uniform<pCN<templateSolver>, templateSolver> MLMCMCSolver(2, 1, 1, 1);
	return 0;
}
