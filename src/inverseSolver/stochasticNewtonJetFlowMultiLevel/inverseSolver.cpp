#include <fstream>
#include <sstream>
// #include <LBFGS.h>
// #include <mpi.h>

// #include <solverInterface.h>
//#include "plainMCMC.cpp"
//#include "stochasticNewtonMCMC.cpp"
// #include <BFGSSolver.h>
#include <MLMCMCSerial.h>
//#include "../mcmcVersions/multiLevelMCMC/oneHierachical/MLMCMC.cpp"
//#include "BiHierachicalMLMCMC.h"


// #include <ChannelCppSolverBiMesh.h>
// #include <jetSolver.h>

// template<typename problemType>
// class forwardProblem{
// public:
// 	problemType turbProp;

// 	~forwardProblem(){};
// 	void init(int level, int validation = 0);
// };

// template<typename problemType>
// void forwardProblem<problemType>::init(int level, int validation){
// 	caseInitialize(turbProp, 550.0, level);
// 	if (validation == 1){
// 		turbProp.validationBoundary();
// 		turbProp.validationPressure();
// 	}
// 	initialization(turbProp); 
// }

// template<typename MCMCType>
// class MCMCMethod{
// private:
// 	MCMCType sampler;

// public:
// 	MCMCMethod(string outputPath);
// 	~MCMCMethod(){};

// 	template<typename problemType>
// 	void run(forwardProblem<problemType>& forwardSolver, double QoI[]);

// 	template<typename problemType>	
// 	void runValidation(forwardProblem<problemType>& forwardSolver, double QoI[], int numCoef_);
// };

// template<typename MCMCType>
// MCMCMethod<MCMCType>::MCMCMethod(string outputPath):sampler(outputPath){}

// template<typename MCMCType>
// template<typename problemType>
// void MCMCMethod<MCMCType>::run(forwardProblem<problemType>& forwardSolver, double QoI[]){
// 	sampler.run(forwardSolver, QoI);
// }

// template<typename MCMCType>
// template<typename problemType>
// void MCMCMethod<MCMCType>::runValidation(forwardProblem<problemType>& forwardSolver, double QoI[], int numCoef_){
// 	sampler.numCoef = numCoef_;
// 	sampler.runValidation(forwardSolver, QoI);
// }



// template<typename MCMCType, typename problemType>
// class inverseSolver{
// private:
// 	MCMCMethod<MCMCType> MCMCSolver;
// 	BFGSSolver<problemType> MAPSolver;

// public:
// 	forwardProblem<problemType> forwardSolver;

// 	inverseSolver(string outputPath);
// 	~inverseSolver(){};

// 	void findMAP(double QoI[]);
// 	void runMCMC(double QoI[]);
// 	void runValidation(double QoI[], int numCoef);
// 	void printBeta();
// };
// template<typename MCMCType, typename problemType>
// inverseSolver<MCMCType, problemType>::inverseSolver(string outputPath):MCMCSolver(outputPath){}

// template<typename MCMCType, typename problemType>
// void inverseSolver<MCMCType, problemType>::runMCMC(double QoI[]){
// 	MCMCSolver.run(forwardSolver, QoI);
// };

// template<typename MCMCType, typename problemType>
// void inverseSolver<MCMCType, problemType>::runValidation(double QoI[], int numCoef){
// 	MCMCSolver.runValidation(forwardSolver, QoI, numCoef);
// };

// template<typename MCMCType, typename problemType>
// void inverseSolver<MCMCType, problemType>::findMAP(double QoI[]){
// 	MAPSolver.findMAP(forwardSolver.turbProp, 10, QoI);
// }

// template<typename MCMCType, typename problemType>
// void inverseSolver<MCMCType, problemType>::printBeta(){
// 	for (int i = 0; i < forwardSolver.turbProp.m; i++){
// 		std::cout << forwardSolver.turbProp.betaML[i].value() << " ";
// 	}	
// }

void generateCaseFolder(string outputPath, char buffer[]){
	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(buffer, 80, "%m%d%H%M", timeinfo);
	outputPath += buffer;
	int error = system(("mkdir " + outputPath).c_str());
	if (error == -1){
		std::cout << "system error" << std::endl;
	}
}

int main(int argc, char **argv){
	//***************************************************
	// int procid, numprocs;
	// MPI_Init(&argc, &argv);
	// MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	// MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	//TODO: Generate Path
	char buffer [80];
	string outputPath = "/home/fenics/share/outputtemp/";
	// if (procid == 0){
	generateCaseFolder(outputPath, buffer);
	// }
	// MPI_Barrier(MPI_COMM_WORLD);
	// MPI_Bcast(buffer, 80, MPI_CHAR, 0, MPI_COMM_WORLD);
	outputPath += buffer;

	int numCoef = stoi(argv[1]);
	int numLevel = stoi(argv[2]);
	int procid = 0;
	// double QoI[2];
	// QoI[0] = 0.2;
	// QoI[1] = 0.2;

	//declare inverse solver
	//********* MCMCType *********** problemType ********
	//
	//1.a        plainMCMC          caseProp<double> 
	// inverseSolver<plainMCMC, caseProp<double>> solver(outputPath);
	// solver.forwardSolver.init(numLevel);
	//solver.findMAP(QoI);
	// solver.runMCMC(QoI);
	//
	//1.b        plainMCMC          caseProp<double>
	// int level = stof(argv[2]);
	// inverseSolver<plainMCMC, caseProp<double>> solver;
	// solver.forwardSolver.init(level);
	// solver.forwardSolver.turbProp.validationBoundary();
	// solver.forwardSolver.turbProp.validationPressure();
	// solver.runValidation(QoI, numCoef);
	//
	//2.   stochasticNewtonMCMC    caseProp<adouble>
	// inverseSolver<stochasticNewtonMCMC, caseProp<adouble>> solver(outputPath);
	// solver.forwardSolver.init(0);
	// solver.runMCMC(QoI);
	//
	//3.a         MLMCMC           caseProp<double>
	// MLMCMC(argc, argv);
	// MLMCMC(procid, numprocs, 550, numCoef, 1);
	//
	//3.b         MLMCMCSerial           caseProp<double>
	MLMCMCSerial<jetFlow>(outputPath, procid, numCoef, numLevel);
	
	//4. Two-Hierachical-MLMCMC
	// double meanError;
	// double workSpace;
	// double sumError;
	// cout << "procid" << procid << endl;
	// BiMLMCMC twoHSolver(outputPath, numLevel, numCoef, 550);

	// workSpace = twoHSolver.run(procid, numprocs);
	// cout << procid << " estimator: " << workSpace << endl << endl;
	// workSpace = abs(workSpace-1);
	// MPI_Reduce(&workSpace, &sumError, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	// MPI_Barrier(MPI_COMM_WORLD);

	// if (procid == 0){
	// 	meanError = sumError/64;
	// 	cout << "meanError" << meanError << endl;		
	// }
	//
	//
	//run solver
	//****Deterministic****
	//


	// solver.printBeta();
	// ****Simulation*******
	
	// solver.runMCMC(QoI);
	// MPI_Finalize();

	return 0;
}
