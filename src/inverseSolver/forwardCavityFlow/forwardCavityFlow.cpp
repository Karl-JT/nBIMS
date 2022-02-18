#include <fstream>
#include <sstream>

#include <cavityFlowCase.h>


int main(int argc, char **argv){

	double cost = 0;
	double gradientVector[2];
	double hessianMatrix[4]; 

	cavityFlowCase cavityFlow;
	cavityFlow.soluFwd();
	cavityFlow.generateRealization();
	cost = cavityFlow.misfit();
	std::cout << "cost: " << cost << std::endl;
	cavityFlow.soluAdj();

	double QoI[2] = {0.1, 0.0};
	cavityFlow.updateParameter(QoI, 2);
	cavityFlow.soluFwd();
	cost = cavityFlow.misfit();
	std::cout << "cost: " << cost << std::endl;
	cavityFlow.soluAdj();
	cavityFlow.grad(gradientVector);
	cavityFlow.hessian(hessianMatrix);

	QoI[0] = 0.2;
	cavityFlow.updateParameter(QoI, 2);
	cavityFlow.soluFwd();
	cost = cavityFlow.misfit();
	std::cout << "cost: " << cost << std::endl;
	cavityFlow.soluAdj();
	cavityFlow.grad(gradientVector);
	cavityFlow.hessian(hessianMatrix);

	// // int numCoef = 2;

	// double QoI[2] = {0.05, 0.0};
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
