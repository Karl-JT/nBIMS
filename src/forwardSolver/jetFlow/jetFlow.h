#include <dolfin.h>
#include "jetFlowSolver.h"
#include <mpi.h>

// int main(int argc, char* argv[]){

// 	dolfin::init(argc, argv);

// 	jetFlowSolver testSolver(30, 15);
// 	double fourierCoef[2] = {0.1, 0.1};

// 	testSolver.updateParameter(fourierCoef);	
// 	testSolver.soluFwd();

// 	testSolver.generateRealization();
// 	testSolver.soluAdj();

// 	double cost;
// 	cost = testSolver.misfit();
// 	cout << "cost: " << cost << endl;

// 	testSolver.grad();
// 	testSolver.hessian();

// 	return 0;
// }

class jetFlow : public jetFlowSolver {
public:
    int procid;
    double cut_off;

    jetFlow(int level) : jetFlowSolver(36*pow(2, level), 12*pow(2, level), level) {
        soluFwd();
    	// uncoupledSolFwd();
    	// updateInitialState();
        // generateRealization();
        generatePointRealization();
    	// soluAdj();
    	// misfit();
    };
    ~jetFlow(){};

    double lnLikelihood(double sample[], int numCoef);
	void adSolver(double sampleProposal[], double gradientVector[], double hessianMatrix[], int numCoef);
};
