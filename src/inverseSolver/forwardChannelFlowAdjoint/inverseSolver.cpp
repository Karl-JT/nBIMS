#include <fstream>
#include <sstream>

#include <turbSolverAdjoint.h>


int main(int argc, char **argv){

	channelFlowAdjoint channelFlow(0);
	channelFlow.soluFwd();
	channelFlow.adSolver();

	return 0;
}
