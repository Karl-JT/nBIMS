#include <string>

#include "MLMCMCChain.h"
#include "stochasticNewtonMCMCChain.h"

template <typename T>
void MLMCMCSerial(string outputPath, int procid_, int numCoef, int numLevel){
	double M;
	// double expect[numCoef] = {0};
	// vector<double> expectList;
	// vector<double> incList;

	// for (int level = 0; level < numLevel; level++){
	// 	if (procid_ == level){
	// 		double singleChainOutput[numCoef] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
	// 		M = 8*pow(2, 2*(numLevel-level));
	// 		MLMCMCChain<stochasticNewtonMCMCChain<T>> singleLevelChain(outputPath, M, procid_, numCoef, level);
	// 		singleLevelChain.run(singleChainOutput);
	// 		for (int i = 0; i < numCoef; i++){
	// 			// expect[i] = expect[i] + singleChainOutput[i];
	// 			// expectList.push_back(expect[i]);
	// 			// incList.push_back(singleChainOutput[i]);
	// 			std::cout << singleChainOutput[i] << " ";
	// 		}
	// 		// std::cout << std::endl;
	// 	}
	// }

	std::string name = "LogProcess";
	std::string path;
	path = name + std::to_string(procid_);

	ofstream myfile;
	myfile.open(path);

	double singleChainOutput[numCoef] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
	M = 10*pow(2, 2*(numLevel-procid_));
	MLMCMCChain<stochasticNewtonMCMCChain<T>> singleLevelChain(outputPath, M, procid_, numCoef, procid_);
	singleLevelChain.run(singleChainOutput);
	for (int i = 0; i < numCoef; i++){
		// expect[i] = expect[i] + singleChainOutput[i];
		// expectList.push_back(expect[i]);
		// incList.push_back(singleChainOutput[i]);
		myfile << singleChainOutput[i] << " ";
	}
	// std::cout << std::endl;

	myfile.close();

	// for (int i  = 0; i < numCoef; i++){
	// 	std::cout << expect[i] << " ";
	// }
	// std::cout << std::endl;
	// for (auto i = expectList.begin(); i!=expectList.end(); ++i){
	// 	std::cout << *i << " ";
	// }
	// std::cout << std::endl;
	// for (auto i = incList.begin(); i!=incList.end(); ++i){
	// 	std::cout << *i << " ";
	// }
}
