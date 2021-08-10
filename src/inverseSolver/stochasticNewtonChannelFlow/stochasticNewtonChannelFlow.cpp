#include <fstream>
#include <sstream>

#include <stochasticNewtonMCMC.h>


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
	string outputPath = "../outputtemp/";
	// if (procid == 0){
	generateCaseFolder(outputPath, buffer);
	//}
	// MPI_Barrier(MPI_COMM_WORLD);
	// MPI_Bcast(buffer, 80, MPI_CHAR, 0, MPI_COMM_WORLD);
	outputPath += buffer;

	int numCoef = stoi(argv[1]);
	int numLevel = stoi(argv[2]);
	int procid = 0;
	//double QoI[2];
	//QoI[0] = 0.2;
	//QoI[1] = 0.2;

	stochasticNewtonMCMC<forwardSolver> inverseSolver(outputPath, procid, numCoef, numLevel);
	inverseSolver.run();

	return 0;
}
