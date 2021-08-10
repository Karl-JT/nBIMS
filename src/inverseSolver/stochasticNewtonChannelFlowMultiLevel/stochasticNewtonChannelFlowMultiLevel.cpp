#include <fstream>
#include <sstream>
// #include <LBFGS.h>
// #include <mpi.h>

// #include <solverInterface.h>
//#include "plainMCMC.cpp"
//#include "stochasticNewtonMCMC.cpp"
// #include <BFGSSolver.h>
#include <MLMCMCSerial.h>

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
	int procid, numprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	//TODO: Generate Path
	char buffer [80];
	string outputPath = "/home/juntao/Desktop/oneDimChanTurbSolver/outputtemp/";
	if (procid == 0){
		generateCaseFolder(outputPath, buffer);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(buffer, 80, MPI_CHAR, 0, MPI_COMM_WORLD);
	outputPath += buffer;

	int numCoef = stoi(argv[1]);
	int numLevel = numprocs; //stoi(argv[2]);
	// int procid = 0;
	// double QoI[2];
	// QoI[0] = 0.2;
	// QoI[1] = 0.2;
	//3.b         MLMCMCSerial           caseProp<double>
	MLMCMCSerial<forwardSolver>(outputPath, procid, numCoef, numLevel);
	//
	//4. Two-Hierachical-MLMCMC
	// double meanError;
	// double workSpace;
	// double sumError;
	// cout << "procid" << procid << endl;
	// BiMLMCMC twoHSolver(outputPath, numLevel, numCoef, 550);


	// solver.printBeta();
	// ****Simulation*******
	
	// solver.runMCMC(QoI);
	MPI_Finalize();

	return 0;
}
