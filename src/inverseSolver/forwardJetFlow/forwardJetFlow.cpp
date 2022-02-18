#include <fstream>
#include <sstream>

#include <jetFlow.h>


int main(int argc, char **argv){

	jetFlow jetFlowSingleLevel(1);

	double gradientVector[2];
	// double hessianMatrix[4]; 
	// int numCoef = 2;

	double QoI[2] = {0.09, 0.0};
	jetFlowSingleLevel.updateParameter(QoI, 2);
	jetFlowSingleLevel.soluFwd();
	// jetFlowSingleLevel.misfit();
	jetFlowSingleLevel.pointwiseMisfit();
	jetFlowSingleLevel.soluAdj();
	jetFlowSingleLevel.grad(gradientVector);
	// jetFlowSingleLevel.hessian(hessianMatrix);

	// jetFlowSingleLevel.updateParameter(QoI, 2);
	// jetFlowSingleLevel.soluFwd();
	// jetFlowSingleLevel.misfit();
	// // jetFlowSingleLevel.pointwiseMisfit();
	// jetFlowSingleLevel.soluAdj();
	// jetFlowSingleLevel.grad(gradientVector);
	// jetFlowSingleLevel.hessian(hessianMatrix);


	// double QoI1[2] = {0.1, 0.0};
	// jetFlowSingleLevel.updateParameter(QoI1, 2);
	// jetFlowSingleLevel.soluFwd();
	// jetFlowSingleLevel.misfit();
	// // jetFlowSingleLevel.pointwiseMisfit();
	// jetFlowSingleLevel.soluAdj();
	// jetFlowSingleLevel.grad(gradientVector);
	// jetFlowSingleLevel.hessian(hessianMatrix);


	// jetFlowSingleLevel.updateParameter(QoI1, 2);
	// jetFlowSingleLevel.soluFwd();
	// jetFlowSingleLevel.misfit();
	// // jetFlowSingleLevel.pointwiseMisfit();
	// jetFlowSingleLevel.soluAdj();
	// jetFlowSingleLevel.grad(gradientVector);
	// jetFlowSingleLevel.hessian(hessianMatrix);


	// double QoI2[2] = {0.05, 0.0};
	// jetFlowSingleLevel.updateParameter(QoI2, 2);
	// jetFlowSingleLevel.soluFwd();
	// jetFlowSingleLevel.misfit();
	// // jetFlowSingleLevel.pointwiseMisfit();
	// jetFlowSingleLevel.soluAdj();
	// jetFlowSingleLevel.grad(gradientVector);
	// jetFlowSingleLevel.hessian(hessianMatrix);

	// jetFlowSingleLevel.updateParameter(QoI2, 2);
	// jetFlowSingleLevel.soluFwd();
	// jetFlowSingleLevel.misfit();
	// // jetFlowSingleLevel.pointwiseMisfit();
	// jetFlowSingleLevel.soluAdj();
	// jetFlowSingleLevel.grad(gradientVector);
	// jetFlowSingleLevel.hessian(hessianMatrix);

	// jetFlowSingleLevel.uncoupledSolFwd();
	// jetFlowSingleLevel.updateInitialState();
	// jetFlowSingleLevel.adSolver(QoI, gradientVector, hessianMatrix, numCoef);
	// jetFlowSingleLevel.misfit();

	// double QoI2[2] = {-0.1, 0.0};
	// jetFlowSingleLevel.updateParameter(QoI2, 2);
	// jetFlowSingleLevel.soluFwd();
	// jetFlowSingleLevel.pointwiseMisfit();
	// jetFlowSingleLevel.soluAdj();
	// jetFlowSingleLevel.grad(gradientVector);
	// jetFlowSingleLevel.hessian(hessianMatrix);

	// double QoI3[2] = {0.0101, 0.0};
	// jetFlowSingleLevel.updateParameter(QoI3, 2);
	// jetFlowSingleLevel.soluFwd();
	// jetFlowSingleLevel.pointwiseMisfit();
	// jetFlowSingleLevel.soluAdj();
	// jetFlowSingleLevel.grad(gradientVector);
	// jetFlowSingleLevel.hessian(hessianMatrix);


	// double QoI4[2] = {0.02, 0.0};
	// jetFlowSingleLevel.updateParameter(QoI4, 2);
	// jetFlowSingleLevel.soluFwd();
	// jetFlowSingleLevel.pointwiseMisfit();
	// jetFlowSingleLevel.soluAdj();
	// jetFlowSingleLevel.grad(gradientVector);
	// jetFlowSingleLevel.hessian(hessianMatrix);

	return 0;
}
