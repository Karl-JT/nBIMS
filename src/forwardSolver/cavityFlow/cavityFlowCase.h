#include <dolfin.h>

#include "cavityFlowSolver.h"
#include <mpi.h>

class cavityFlowCase : public cavityFlowSolver {
public:
	cavityFlowCase() : cavityFlowSolver(32, 32){};
	~cavityFlowCase(){};
};