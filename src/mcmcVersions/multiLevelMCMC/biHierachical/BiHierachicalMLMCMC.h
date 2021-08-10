#include "../../../forwardSolver/bisectionMesh/ChannelCppSolverBiMesh.h" 
#include "../../mcmcCore/MCMCMetropolis.h"
#include "../../mcmcCore/MLMCMCChain.h"

#include <mpi.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <memory>

//#include <gsl/gsl_rng.h>
//#include <gsl/gsl_randist.h>
//#include <gsl/gsl_cdf.h>
//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_linalg.h>
#include <random>
#include <time.h>


class BiMLMCMC{
private:
public:
	string outputPath;

	int numLevel;
	int numCoef;

	double reTau;
	double finalSum = 0;
	double finalExpectation = 0;

	int randomSeed = time(NULL);

	default_random_engine generator;

	//gsl_rng *r;
	//const gsl_rng_type *T;

	BiMLMCMC(string outputPath_, int numLevel_, int numCoef_, double reTau_);
	~BiMLMCMC(){};

	double runOneChain(string outputPath,int numLevel, int sampleL, int solverL, int chainID, double reTau, int numCoef, int procid, bool validation);
	double run(int procid, int numprocs);
};
