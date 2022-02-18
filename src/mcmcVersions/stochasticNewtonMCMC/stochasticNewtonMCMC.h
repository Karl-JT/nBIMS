#include <stochasticNewtonMCMCChain.h>

template <typename T>
class stochasticNewtonMCMC{
public:
	int dimension;
	double M = 2000;
	std::unique_ptr<double[]> startPointArray;
	std::shared_ptr<stochasticNewtonMCMCChain<T>> singleLevelChain;

	stochasticNewtonMCMC(string outputPath, int procid_, int numCoef, int level);
	~stochasticNewtonMCMC(){};

	void startPoint(double QoI[]);
	void burnin();
	void run();
};

template <typename T>
stochasticNewtonMCMC<T>::stochasticNewtonMCMC(string outputPath, int procid_, int numCoef, int level){
	dimension = numCoef;
	singleLevelChain = make_shared<stochasticNewtonMCMCChain<T>>(outputPath, M, procid_, numCoef, level);
	startPointArray = std::make_unique<double[]>(numCoef);
	for (int i = 0; i < dimension; ++i){
		startPointArray[i] = 0.2;
	}
}

template <typename T>
void stochasticNewtonMCMC<T>::startPoint(double QoI[]){
	for (int i = 0; i < dimension; ++i){
		startPointArray[i] = QoI[i];
	}
}

template <typename T>
void stochasticNewtonMCMC<T>::burnin(){
	singleLevelChain->startPoint(startPointArray.get());
	singleLevelChain->RWSwitch = 0;
	for (int i = 0; i < 25; i++){
		singleLevelChain->runStep();
		singleLevelChain->samplerSolver->cut_off = max(1.0, (singleLevelChain->samplerSolver->cut_off)*0.5);
	}
	singleLevelChain->samplerSolver->cut_off = 1.0;//-2000.0*singleLevelChain->samplerLevel;
	singleLevelChain->RWSwitch = 1;
	singleLevelChain->runStep();
	singleLevelChain->startPoint(singleLevelChain->sampleProposal.get());

	singleLevelChain->runStep();
}


template <typename T>
void stochasticNewtonMCMC<T>::run(){
	burnin();
	singleLevelChain->run(singleLevelChain->sampleCurrent.get());
}
