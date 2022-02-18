#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <memory>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <petsc.h>

PetscErrorCode MyKSPMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *dummy)
{
	PetscPrintf(PETSC_COMM_SELF, "iteration %D KSP Residual norm %14.12e \n", n, rnorm);
	return 0;
}

class stokesSolver {
private:
	std::unique_ptr<std::unique_ptr<int[]>[]> quadrilaterals;
	std::unique_ptr<int[]> Q1_idx;
	std::unique_ptr<int[]> mesh_idx;
	std::unique_ptr<int[]> Bs;

	Mat A1;
	Mat A2;
	Mat B1;
	Mat B1T;
	Mat B2;
	Mat B2T;
	Mat D;
	Mat I;
	Mat sysMatrix;
	Mat finalMatrix;
	Mat state2obsMatrix;
	Vec V;
	Vec rhs;
	Vec adjointRhs;
	Vec finalRhs;

	KSP ksp;
	PC  pc;

public:
	Vec states;
	Vec adjoints;

	int division0 = 2;
	int division_Q1isoQ2;
	int division_Q1;
	int num_Q1isoQ2_element;
	int num_Q1_element;
	int num_node_Q1isoQ2;
	int num_node_Q1;

	std::unique_ptr<std::unique_ptr<double[]>[]> points;

	int num_term;
	int level;
	double noiseVariance;
	double beta = 1;
	double *states_array;
	double *adjoint_array;
	std::unique_ptr<double[]> u;
	std::unique_ptr<double[]> u1;
	std::unique_ptr<double[]> u2;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution{0.0, 1.0};

	stokesSolver(int level_, int num_term_, double noiseVariance_);
	~stokesSolver(){};

	double fsource(double x, double y);
	double fsource_i(double x, double y, int i);
	void shape_function(double epsilon, double eta, double N[]);
	double shape_interpolation(double epsilon, double eta, double x[4]);
	void jacobian_matrix(double x1[4], double x2[4], double epsilon, double eta, double J[2][2]);
	double jacobian_det(double J[2][2]);
	void jacobian_inv(double x1[4], double x2[4], double epsilon, double eta, double J[2][2]);
	void basis_function(double epsilon, double eta, double N[]);
	void basis_function_derivative(double dPhi[4][2], double epsilon, double eta);
	void hat_function_derivative(double dPhi[4][2], double epsilon, double eta, double x1[4], double x2[4]);
	void stiffness_matrix_element(double P[][4], double A[][4]);
	void stiffness_matrix();
	void b1_matrix_element(double x1[4], double x2[4], double b1[4][4]);
	void b2_matrix_element(double x1[4], double x2[4], double b1[4][4]);
	void B1_matrix();
	void B2_matrix();
	void system_matrix();
	void Q1isoQ2Cond();
	void load_vector_element(double x1[], double x2[], double v[]);
	void area_vector_element();
	void load_vector();
	void load_vector_element_i(double x1[], double x2[], double v[], int i);
	void load_vector_i(int i, double load[]);
	void system_rhs();
	void apply_boundary_condition();
	void apply_homo_boundary_condition();
	void linear_system_setup();
	void pointObsMatrix(double obs[][4], int size);
	void forwardEqn();
	void adjointEqn(int size, double obs[][4]);
	void controlEqn(double grad[]);	
	void WuuApply(Vec direction, int size);
	Vec CApply(Vec direction, int size, bool T);
	Vec ASolve(Vec direction, int size, bool T);
	Vec RApply(Vec direction, bool T);
	void HessianAction(double grad[]);
	void fullHessian();
	void updateGeneratorSeed(double seed_);
	void pCNProposalGenerator(double proposal[]);
	void SNProposalGenerator(double proposal[]);
	double lnlikelihood(double obs[][4], int size);
	double solve(double obs[][4], int size);
};

stokesSolver::stokesSolver(int level_, int num_term_, double noiseVariance_) : level(level_), num_term(num_term_), noiseVariance(noiseVariance_) {
	u = std::make_unique<double[]>(num_term_);

	division_Q1isoQ2 = division0*(std::pow(2, level_+1));
	division_Q1 = division_Q1isoQ2/2;
	num_Q1isoQ2_element = division_Q1isoQ2*division_Q1isoQ2;
	num_Q1_element = 0.25*num_Q1isoQ2_element;
	num_node_Q1isoQ2 = std::pow(division_Q1isoQ2+1, 2);
	num_node_Q1 = std::pow(division_Q1+1, 2);

	points    = std::make_unique<std::unique_ptr<double[]>[]>(2);
	points[0] = std::make_unique<double[]>(num_node_Q1isoQ2);
	points[1] = std::make_unique<double[]>(num_node_Q1isoQ2);

	double xCoord = 0;
	double yCoord = 0;
	for (int i=0; i<num_node_Q1isoQ2; i++){
		if (xCoord-1 > 1e-6){
			xCoord = 0;
			yCoord += 1.0/division_Q1isoQ2;
		}
		points[0][i] = xCoord;
		points[1][i] = yCoord;
		xCoord += 1.0/division_Q1isoQ2;
	}

	quadrilaterals = std::make_unique<std::unique_ptr<int[]>[]>(4);
	for (int i=0; i<4; i++){
		quadrilaterals[i] = std::make_unique<int[]>(num_Q1isoQ2_element);
	}
	int refDof[4] = {0, 1, division_Q1isoQ2+2, division_Q1isoQ2+1};
	for (int i=0; i<num_Q1isoQ2_element; i++){
		quadrilaterals[0][i] = refDof[0];
		quadrilaterals[1][i] = refDof[1];
		quadrilaterals[2][i] = refDof[2];
		quadrilaterals[3][i] = refDof[3];

		if ((refDof[1]+1)%(division_Q1isoQ2+1) == 0){
			refDof[0] += 2;
			refDof[1] += 2;
			refDof[2] += 2;
			refDof[3] += 2;
		} else {
			refDof[0] += 1;
			refDof[1] += 1;
			refDof[2] += 1;
			refDof[3] += 1;
		}
	}

	Q1_idx = std::make_unique<int[]>(num_node_Q1isoQ2);
	for (int i = 0; i < num_node_Q1isoQ2; i++){
		Q1_idx[i] = num_node_Q1isoQ2;
	}

	int position = 0;
	int value = 0;
	for (int i = 0; i < division_Q1isoQ2/2.0+1; i++){
		position = 2*(division_Q1isoQ2+1)*i;
		for (int j = 0; j < division_Q1isoQ2/2.0+1; j++){
			Q1_idx[position] = value;
			value += 1; 
			position = position + 2;
		}
	}

	for (int i = 0; i < num_node_Q1isoQ2; i++){
		if (Q1_idx[i] == num_node_Q1isoQ2){
			Q1_idx[i] += i;
		}
	}

	std::vector<int> mesh_idx_vector(num_node_Q1isoQ2);
	std::vector<int> mesh_idx_vector2(num_node_Q1isoQ2);

	std::iota(mesh_idx_vector.begin(), mesh_idx_vector.end(), 0);
	std::iota(mesh_idx_vector2.begin(), mesh_idx_vector2.end(), 0);

	std::stable_sort(mesh_idx_vector.begin(), mesh_idx_vector.end(), [&](int i, int j){return Q1_idx[i] < Q1_idx[j];});
	std::stable_sort(mesh_idx_vector2.begin(), mesh_idx_vector2.end(), [&](int i, int j){return mesh_idx_vector[i] < mesh_idx_vector[j];});
	mesh_idx = std::make_unique<int[]>(num_node_Q1isoQ2);
	for (int i = 0; i < num_node_Q1isoQ2; i++){
		mesh_idx[i] = mesh_idx_vector2[i];
	}

	int size = division_Q1isoQ2*4;
	Bs = std::make_unique<int[]>(2*size);
	int b_pos[4] = {0, division_Q1isoQ2, num_node_Q1isoQ2-1, num_node_Q1isoQ2-1-division_Q1isoQ2};	 
	for (int i=0; i<division_Q1isoQ2; i++){
		Bs[i]                    = b_pos[0]+i;
		Bs[i+division_Q1isoQ2]   = b_pos[1]+(division_Q1isoQ2+1)*i;
		Bs[i+2*division_Q1isoQ2] = b_pos[2]-i;
		Bs[i+3*division_Q1isoQ2] = b_pos[3]-(division_Q1isoQ2+1)*i;
	}
	for (int i = 0; i < 4*division_Q1isoQ2; i++){
		Bs[i+4*division_Q1isoQ2] +=  Bs[i]+num_node_Q1isoQ2;
	}

	linear_system_setup();
	system_rhs();
	apply_boundary_condition();

	KSPCreate(PETSC_COMM_SELF, &ksp);
	KSPGMRESSetRestart(ksp, 100);
	KSPSetOperators(ksp, finalMatrix, finalMatrix);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCJACOBI);
	KSPSetTolerances(ksp, 1e-3, 1e-6, PETSC_DEFAULT, 100000);
	// KSPMonitorSet(ksp, MyKSPMonitor, NULL, 0);
	KSPSetUp(ksp);

	u1 = std::make_unique<double[]>(num_node_Q1isoQ2);
	u2 = std::make_unique<double[]>(num_node_Q1isoQ2);
};

void stokesSolver::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};

double stokesSolver::fsource(double x, double y){
    double output = 0;
    output = u[0]*std::sin(2.*M_PI*x)*std::sin(2.*M_PI*y) \
           + u[1]*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y) \
           + u[2]*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y) \
           + u[3]*std::cos(2.*M_PI*x)*std::cos(2.*M_PI*y) ;
    return 100*output;
}

double stokesSolver::fsource_i(double x, double y, int m_idx){
    double output;
    double indicator[4] = {0.0};    
    indicator[m_idx] = 1.0;
    output = indicator[0]*std::sin(2.*M_PI*x)*std::sin(2.*M_PI*y) \
           + indicator[1]*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y) \
           + indicator[2]*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y) \
           + indicator[3]*std::cos(2.*M_PI*x)*std::cos(2.*M_PI*y) ;
    return 100*output;
}

void stokesSolver::shape_function(double epsilon, double eta, double N[]){
	N[0] = 0.25*(1.0-epsilon)*(1.0-eta);
	N[1] = 0.25*(1.0+epsilon)*(1.0-eta);
	N[2] = 0.25*(1.0+epsilon)*(1.0+eta);
	N[3] = 0.25*(1.0-epsilon)*(1.0+eta);
};

double stokesSolver::shape_interpolation(double epsilon, double eta, double x[4]){
	double N[4];
	double x_interpolated;
	shape_function(epsilon, eta, N);
	x_interpolated = N[0]*x[0]+N[1]*x[1]+N[2]*x[2]+N[3]*x[3];
	return x_interpolated;
}

void stokesSolver::jacobian_matrix(double x1[4], double x2[4], double eta, double epsilon, double J[2][2]){
	J[0][0] = 0.25*((eta-1)*x1[0]+(1-eta)*x1[1]+(1+eta)*x1[2]-(1+eta)*x1[3]); 
	J[0][1] = 0.25*((eta-1)*x2[0]+(1-eta)*x2[1]+(1+eta)*x2[2]-(1+eta)*x2[3]); 
	J[1][0] = 0.25*((epsilon-1)*x1[0]-(1+epsilon)*x1[1]+(1+epsilon)*x1[2]+(1-epsilon)*x1[3]); 
	J[1][1] = 0.25*((epsilon-1)*x2[0]-(1+epsilon)*x2[1]+(1+epsilon)*x2[2]+(1-epsilon)*x2[3]); 
};

double stokesSolver::jacobian_det(double J[2][2]){
	double detJ = J[1][1]*J[0][0] - J[0][1]*J[1][0];
	return detJ;
};

void stokesSolver::jacobian_inv(double x1[4], double x2[4], double epsilon, double eta, double Jinv[2][2]){
	double J[2][2];
	jacobian_matrix(x1, x2, epsilon, eta, J);
	double Jdet = jacobian_det(J);
	Jinv[0][0] = J[1][1]/Jdet;	
	Jinv[0][1] =-J[0][1]/Jdet;
	Jinv[1][0] =-J[1][0]/Jdet;	
	Jinv[1][1] = J[0][0]/Jdet;
};

void stokesSolver::basis_function(double epsilon, double eta, double N[]){
	shape_function(epsilon, eta, N);
};

void stokesSolver::basis_function_derivative(double basisdPhi[4][2], double epsilon, double eta){
	basisdPhi[0][0] = -0.25+0.25*eta;	
	basisdPhi[1][0] = 0.25-0.25*eta;
	basisdPhi[2][0] = 0.25+0.25*eta;
	basisdPhi[3][0] = -0.25-0.25*eta;
	basisdPhi[0][1] = -0.25+0.25*epsilon;	
	basisdPhi[1][1] = -0.25-0.25*epsilon;
	basisdPhi[2][1] = 0.25+0.25*epsilon;
	basisdPhi[3][1] = 0.25-0.25*epsilon;		
};

void stokesSolver::hat_function_derivative(double dPhi[4][2], double epsilon, double eta, double x1[4], double x2[4]){
	double basisdPhi[4][2];
	double Jinv[2][2];
	basis_function_derivative(basisdPhi, epsilon, eta);
	jacobian_inv(x1, x2, epsilon, eta, Jinv);
	for (int i = 0; i < 4; ++i){
		for (int j = 0; j < 2; ++j){
			dPhi[i][j] = basisdPhi[i][0]*Jinv[0][j] + basisdPhi[i][1]*Jinv[1][j];
		}
	}
}

void stokesSolver::stiffness_matrix_element(double P[][4], double A[][4]){
	double dPhi[4][2];
	double Jinv[2][2];
	double J[2][2];

	double refPoints[5];
	double refWeights[5];

	refPoints[0] = -1./3.*sqrt(5.+2.*sqrt(10./7.));
	refPoints[1] = -1./3.*sqrt(5.-2.*sqrt(10./7.));
	refPoints[2] = 0.;
	refPoints[3] = 1./3.*sqrt(5.-2.*sqrt(10./7.));
	refPoints[4] = 1./3.*sqrt(5.+2.*sqrt(10./7.));

	refWeights[0] = (322.-13.*sqrt(70.))/900.;
	refWeights[1] = (322.+13.*sqrt(70.))/900.;
	refWeights[2] = 128./225.;
	refWeights[3] = (322.+13.*sqrt(70.))/900.;
	refWeights[4] = (322.-13.*sqrt(70.))/900.;

	double gpPoints[25][8];
	double gpWeights[25];
	double Jdet[25];

	for (int i = 0; i < 5; i++){
		for (int j = 0; j < 5; j++){
			hat_function_derivative(dPhi, refPoints[i], refPoints[j], P[0], P[1]);
			jacobian_matrix(P[0], P[1], refPoints[i], refPoints[j], J);
			gpWeights[5*i+j] = refWeights[i]*refWeights[j];
			Jdet[5*i+j] = jacobian_det(J);
			for (int k = 0; k < 8; ++k){
				gpPoints[5*i+j][k] = dPhi[k%4][k/4];
			}
		}
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			A[i][j] = 0;
			for (int k = 0; k < 25; k++){
				A[i][j] += gpWeights[k]*(gpPoints[k][i]*gpPoints[k][j]+gpPoints[k][4+i]*gpPoints[k][4+j])*Jdet[k];
			}
		}
	}
};

void stokesSolver::stiffness_matrix(){
	MatCreate(PETSC_COMM_SELF, &A1);
	MatSetSizes(A1, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	MatSetFromOptions(A1);
	MatSeqAIJSetPreallocation(A1, 12, NULL);
	MatMPIAIJSetPreallocation(A1, 12, NULL, 12, NULL);
	double stiffness_element[4][4];
	double element_points[2][4];
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		stiffness_matrix_element(element_points, stiffness_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
 				MatSetValue(A1, quadrilaterals[m][i], quadrilaterals[n][i], stiffness_element[m][n], ADD_VALUES);
			}
		}			
	}
	MatAssemblyBegin(A1, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A1, MAT_FINAL_ASSEMBLY);

	MatCreate(PETSC_COMM_SELF, &A2);
	MatSetSizes(A2, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	MatSetFromOptions(A2);
	MatSeqAIJSetPreallocation(A2, 12, NULL);
	MatMPIAIJSetPreallocation(A2, 12, NULL, 12, NULL);
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		stiffness_matrix_element(element_points, stiffness_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
 				MatSetValue(A2, num_node_Q1isoQ2+quadrilaterals[m][i], num_node_Q1isoQ2+quadrilaterals[n][i], stiffness_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(A2, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A2, MAT_FINAL_ASSEMBLY);
};

void stokesSolver::b1_matrix_element(double x1[4], double x2[4], double b1[4][4]){
	double dPhi[4][2];
	double Jinv[2][2];
	double J[2][2];
	double N[4];

	double refPoints[5];
	double refWeights[5];

	refPoints[0] = -1./3.*sqrt(5.+2.*sqrt(10./7.));
	refPoints[1] = -1./3.*sqrt(5.-2.*sqrt(10./7.));
	refPoints[2] = 0.;
	refPoints[3] = 1./3.*sqrt(5.-2.*sqrt(10./7.));
	refPoints[4] = 1./3.*sqrt(5.+2.*sqrt(10./7.));

	refWeights[0] = (322.-13.*sqrt(70.))/900.;
	refWeights[1] = (322.+13.*sqrt(70.))/900.;
	refWeights[2] = 128./225.;
	refWeights[3] = (322.+13.*sqrt(70.))/900.;
	refWeights[4] = (322.-13.*sqrt(70.))/900.;

	double gpPoints1[25][4];
	double gpPoints2[25][4];
	double gpWeights[25];
	double Jdet[25];

	for (int i = 0; i < 5; i++){
		for (int j = 0; j < 5; j++){
			basis_function(refPoints[i], refPoints[j], N);
			hat_function_derivative(dPhi, refPoints[i], refPoints[j], x1, x2);
			jacobian_matrix(x1, x2, refPoints[i], refPoints[j], J);
			gpWeights[5*i+j] = refWeights[i]*refWeights[j];
			Jdet[5*i+j] = jacobian_det(J);
			for (int k = 0; k < 4; ++k){
				gpPoints1[5*i+j][k] = dPhi[k][0];
				gpPoints2[5*i+j][k] = N[k];
			}
		}
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			b1[i][j] = 0;
			for (int k = 0; k < 25; k++){
				b1[i][j] += gpWeights[k]*(gpPoints1[k][j]*gpPoints2[k][i])*Jdet[k];
			}
		}
	}
};

void stokesSolver::b2_matrix_element(double x1[4], double x2[4], double b2[4][4]){
	double dPhi[4][2];
	double Jinv[2][2];
	double J[2][2];
	double N[4];

	double refPoints[5];
	double refWeights[5];

	refPoints[0] = -1./3.*sqrt(5.+2.*sqrt(10./7.));
	refPoints[1] = -1./3.*sqrt(5.-2.*sqrt(10./7.));
	refPoints[2] = 0.;
	refPoints[3] = 1./3.*sqrt(5.-2.*sqrt(10./7.));
	refPoints[4] = 1./3.*sqrt(5.+2.*sqrt(10./7.));

	refWeights[0] = (322.-13.*sqrt(70.))/900.;
	refWeights[1] = (322.+13.*sqrt(70.))/900.;
	refWeights[2] = 128./225.;
	refWeights[3] = (322.+13.*sqrt(70.))/900.;
	refWeights[4] = (322.-13.*sqrt(70.))/900.;

	double gpPoints1[25][4];
	double gpPoints2[25][4];
	double gpWeights[25];
	double Jdet[25];

	for (int i = 0; i < 5; i++){
		for (int j = 0; j < 5; j++){
			basis_function(refPoints[i], refPoints[j], N);
			hat_function_derivative(dPhi, refPoints[i], refPoints[j], x1, x2);
			jacobian_matrix(x1, x2, refPoints[i], refPoints[j], J);
			gpWeights[5*i+j] = refWeights[i]*refWeights[j];
			Jdet[5*i+j] = jacobian_det(J);
			for (int k = 0; k < 4; ++k){
				gpPoints1[5*i+j][k] = dPhi[k][1];
				gpPoints2[5*i+j][k] = N[k];
			}
		}
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			b2[i][j] = 0;
			for (int k = 0; k < 25; k++){
				b2[i][j] += gpWeights[k]*(gpPoints1[k][j]*gpPoints2[k][i])*Jdet[k];
			}
		}
	}
};

void stokesSolver::B1_matrix(){
	MatCreate(PETSC_COMM_SELF, &B1);
	MatSetSizes(B1, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	MatSetFromOptions(B1);
	MatSeqAIJSetPreallocation(B1, 12, NULL);
	MatMPIAIJSetPreallocation(B1, 12, NULL, 12, NULL);
	double b1_element[4][4];
	double element_points[2][4];
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		b1_matrix_element(element_points[0], element_points[1], b1_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
				MatSetValue(B1, 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[m][i]], quadrilaterals[n][i], b1_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(B1, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B1, MAT_FINAL_ASSEMBLY);	


	MatCreate(PETSC_COMM_SELF, &B1T);
	MatSetSizes(B1T, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	MatSetFromOptions(B1T);
	MatSeqAIJSetPreallocation(B1T, 12, NULL);
	MatMPIAIJSetPreallocation(B1T, 12, NULL, 12, NULL);
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		b1_matrix_element(element_points[0], element_points[1], b1_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
				MatSetValue(B1T, quadrilaterals[n][i], 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[m][i]], b1_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(B1T, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B1T, MAT_FINAL_ASSEMBLY);	
};

void stokesSolver::B2_matrix(){
	MatCreate(MPI_COMM_SELF, &B2);
	MatSetSizes(B2, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	MatSetFromOptions(B2);
	MatSeqAIJSetPreallocation(B2, 12, NULL);
	MatMPIAIJSetPreallocation(B2, 12, NULL, 12, NULL);
	double b2_element[4][4];
	double element_points[2][4];
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		b2_matrix_element(element_points[0], element_points[1], b2_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
				MatSetValue(B2, 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[m][i]], num_node_Q1isoQ2+quadrilaterals[n][i], b2_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(B2, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B2, MAT_FINAL_ASSEMBLY);	

	MatCreate(MPI_COMM_SELF, &B2T);
	MatSetSizes(B2T, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	MatSetFromOptions(B2T);
	MatSeqAIJSetPreallocation(B2T, 12, NULL);
	MatMPIAIJSetPreallocation(B2T, 12, NULL, 12, NULL);
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}

		b2_matrix_element(element_points[0], element_points[1], b2_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
				MatSetValue(B2T, num_node_Q1isoQ2+quadrilaterals[n][i], 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[m][i]], b2_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(B2T, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B2T, MAT_FINAL_ASSEMBLY);	
};

void stokesSolver::system_matrix(){
	MatAXPY(A1, 1, A2, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(A1, 1, B1, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(A1, 1, B1T, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(A1, 1, B2, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(A1, 1, B2T, DIFFERENT_NONZERO_PATTERN);
};

void stokesSolver::Q1isoQ2Cond(){
	MatCreate(MPI_COMM_SELF, &D);
	MatSetSizes(D, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	MatSetFromOptions(D);
	MatSeqAIJSetPreallocation(D, 12, NULL);
	MatMPIAIJSetPreallocation(D, 12, NULL, 12, NULL);
	for (int i = 0; i < 2*num_node_Q1isoQ2; i++){
		MatSetValue(D, i, i, 1, INSERT_VALUES);
	}
	for (int i = 0; i < num_node_Q1; i++){
		MatSetValue(D, 2*num_node_Q1isoQ2+i, 2*num_node_Q1isoQ2+i, 1, INSERT_VALUES);
	}
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		if (Q1_idx[quadrilaterals[0][i]] < num_node_Q1isoQ2){
			MatSetValue(D, 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[1][i]], 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[0][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[3][i]], 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[0][i]], 0.5, INSERT_VALUES);
		} else if (Q1_idx[quadrilaterals[1][i]] < num_node_Q1isoQ2){
			MatSetValue(D, 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[2][i]], 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[1][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[0][i]], 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[1][i]], 0.5, INSERT_VALUES);			
		} else if (Q1_idx[quadrilaterals[2][i]] < num_node_Q1isoQ2){
			MatSetValue(D, 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[3][i]], 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[2][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[1][i]], 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[2][i]], 0.5, INSERT_VALUES);			
		} else if (Q1_idx[quadrilaterals[3][i]] < num_node_Q1isoQ2){
			MatSetValue(D, 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[0][i]], 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[3][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[2][i]], 2*num_node_Q1isoQ2+mesh_idx[quadrilaterals[3][i]], 0.5, INSERT_VALUES);			
		}
	}
	MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);

	Mat workspace;	
	MatMatMult(A1, D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &workspace);
	MatTransposeMatMult(D, workspace, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &sysMatrix);

	IS isrow;
	IS iscol;
	int final_size = 2*num_node_Q1isoQ2+num_node_Q1;
	PetscInt *indices;
	PetscMalloc1(final_size, &indices);
	for (int i = 0; i < final_size; i++){
		indices[i] = i;
	}
	ISCreateGeneral(PETSC_COMM_SELF, final_size, indices, PETSC_COPY_VALUES, &isrow);
	ISCreateGeneral(PETSC_COMM_SELF, final_size, indices, PETSC_COPY_VALUES, &iscol);
	MatCreateSubMatrix(sysMatrix, isrow, iscol, MAT_INITIAL_MATRIX, &finalMatrix);
};

void stokesSolver::load_vector_element(double x1[], double x2[], double v[]){
	double J[2][2];

	double refPoints[5];
	double refWeights[5];

	refPoints[0] = -1./3.*sqrt(5.+2.*sqrt(10./7.));
	refPoints[1] = -1./3.*sqrt(5.-2.*sqrt(10./7.));
	refPoints[2] = 0.;
	refPoints[3] = 1./3.*sqrt(5.-2.*sqrt(10./7.));
	refPoints[4] = 1./3.*sqrt(5.+2.*sqrt(10./7.));

	refWeights[0] = (322.-13.*sqrt(70.))/900.;
	refWeights[1] = (322.+13.*sqrt(70.))/900.;
	refWeights[2] = 128./225.;
	refWeights[3] = (322.+13.*sqrt(70.))/900.;
	refWeights[4] = (322.-13.*sqrt(70.))/900.;

	double gpPoints1[25];
	double gpPoints2[25][4];
	double gpWeights[25];
	double Jdet[25];
	double N[4];

	double x;
	double y;

	for (int i = 0; i < 5; i++){
		for (int j = 0; j < 5; j++){
			basis_function(refPoints[i], refPoints[j], N);
			jacobian_matrix(x1, x2, refPoints[i], refPoints[j], J);
			x = shape_interpolation(refPoints[i], refPoints[j], x1);
			y = shape_interpolation(refPoints[i], refPoints[j], x2);
			gpWeights[5*i+j] = refWeights[i]*refWeights[j];
			Jdet[5*i+j] = jacobian_det(J);
			gpPoints1[5*i+j] = fsource(x, y);
			for (int k = 0; k < 4; ++k){
				gpPoints2[5*i+j][k] = N[k];
			}
		}
	}
	for (int i = 0; i < 4; i++){
		v[i] = 0;
		for (int k = 0; k < 25; k++){
			v[i] += gpWeights[k]*gpPoints1[k]*gpPoints2[k][i]*Jdet[k];
		}
	}
};

void stokesSolver::load_vector(){
 	VecCreate(MPI_COMM_SELF, &V);
	VecSetSizes(V, PETSC_DECIDE, 3*num_node_Q1isoQ2);
	VecSetFromOptions(V);
	double v_element[4];
	double element_points[2][4];
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		load_vector_element(element_points[0], element_points[1], v_element);
		for (int k = 0; k < 4; k++){
			VecSetValue(V, quadrilaterals[k][i], v_element[k], ADD_VALUES);
			VecSetValue(V, num_node_Q1isoQ2+quadrilaterals[k][i], v_element[k], ADD_VALUES);
		}
	}
	VecAssemblyBegin(V);
	VecAssemblyEnd(V);
};



void stokesSolver::load_vector_element_i(double x1[], double x2[], double v[], int i){
	double J[2][2];

	double refPoints[5];
	double refWeights[5];

	refPoints[0] = -1./3.*sqrt(5.+2.*sqrt(10./7.));
	refPoints[1] = -1./3.*sqrt(5.-2.*sqrt(10./7.));
	refPoints[2] = 0.;
	refPoints[3] = 1./3.*sqrt(5.-2.*sqrt(10./7.));
	refPoints[4] = 1./3.*sqrt(5.+2.*sqrt(10./7.));

	refWeights[0] = (322.-13.*sqrt(70.))/900.;
	refWeights[1] = (322.+13.*sqrt(70.))/900.;
	refWeights[2] = 128./225.;
	refWeights[3] = (322.+13.*sqrt(70.))/900.;
	refWeights[4] = (322.-13.*sqrt(70.))/900.;

	double gpPoints1[25];
	double gpPoints2[25][4];
	double gpWeights[25];
	double Jdet[25];
	double N[4];

	double x;
	double y;

	for (int i = 0; i < 5; i++){
		for (int j = 0; j < 5; j++){
			basis_function(refPoints[i], refPoints[j], N);
			jacobian_matrix(x1, x2, refPoints[i], refPoints[j], J);
			x = shape_interpolation(refPoints[i], refPoints[j], x1);
			y = shape_interpolation(refPoints[i], refPoints[j], x2);
			gpWeights[5*i+j] = refWeights[i]*refWeights[j];
			Jdet[5*i+j] = jacobian_det(J);
			gpPoints1[5*i+j] = fsource_i(x, y, i);
			for (int k = 0; k < 4; ++k){
				gpPoints2[5*i+j][k] = N[k];
			}
		}
	}
	for (int i = 0; i < 4; i++){
		v[i] = 0;
		for (int k = 0; k < 25; k++){
			v[i] += gpWeights[k]*gpPoints1[k]*gpPoints2[k][i]*Jdet[k];
		}
	}
}

void stokesSolver::load_vector_i(int i, double load[]){
 	VecCreate(MPI_COMM_SELF, &V);
	VecSetSizes(V, PETSC_DECIDE, 3*num_node_Q1isoQ2);
	VecSetFromOptions(V);
	double v_element[4];
	double element_points[2][4];
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		load_vector_element_i(element_points[0], element_points[1], v_element, i);
		for (int k = 0; k < 4; k++){
			VecSetValue(V, quadrilaterals[k][i], v_element[k], ADD_VALUES);
			VecSetValue(V, num_node_Q1isoQ2+quadrilaterals[k][i], v_element[k], ADD_VALUES);
		}
	}
	VecAssemblyBegin(V);
	VecAssemblyEnd(V);
};


void stokesSolver::system_rhs(){
	load_vector();

 	VecCreate(MPI_COMM_SELF, &rhs);
	VecSetSizes(rhs, PETSC_DECIDE, 3*num_node_Q1isoQ2);
	VecSetFromOptions(rhs);
	VecCopy(V, rhs);

	IS isVector;
	int final_size = 2*num_node_Q1isoQ2+num_node_Q1;
	PetscInt *indices;
	PetscMalloc1(final_size, &indices);
	for (int i = 0; i < final_size; i++){
		indices[i] = i;
	}

	ISCreateGeneral(PETSC_COMM_SELF, final_size, indices, PETSC_COPY_VALUES, &isVector);
	VecGetSubVector(rhs, isVector, &finalRhs);

	VecDuplicate(finalRhs, &states);
};

void stokesSolver::apply_boundary_condition(){
	Vec workspace;
	VecDuplicate(states, &workspace);
	VecSet(workspace, 0.0);
	VecAssemblyBegin(workspace);
	VecAssemblyEnd(workspace);
	MatZeroRows(finalMatrix, 2*4*division_Q1isoQ2, Bs.get(), 1.0, workspace, finalRhs);
	// MatZeroRowsColumns(finalMatrix, 2*4*division_Q1isoQ2, Bs.get(), 1.0, states, finalRhs);
};


void stokesSolver::apply_homo_boundary_condition(){
	Vec workspace;
	VecDuplicate(states, &workspace);
	VecSet(workspace, 0.0);
	VecAssemblyBegin(workspace);
	VecAssemblyEnd(workspace);
	MatZeroRows(finalMatrix, 2*4*division_Q1isoQ2, Bs.get(), 1.0, workspace, adjointRhs);
	// MatZeroRowsColumns(finalMatrix, 2*4*division_Q1isoQ2, Bs.get(), 1.0, states, finalRhs);
};


void stokesSolver::linear_system_setup(){
	stiffness_matrix();
	B1_matrix();
	B2_matrix();
	system_matrix();
	Q1isoQ2Cond();	
}

void stokesSolver::forwardEqn(){
	system_rhs();
	apply_boundary_condition();

	KSPSolve(ksp, finalRhs, states);
	VecGetArray(states, &states_array);
	std::copy(states_array, states_array+num_node_Q1isoQ2, u1.get());
	std::copy(states_array+num_node_Q1isoQ2, states_array+2*num_node_Q1isoQ2, u2.get());
	// VecView(states, PETSC_VIEWER_STDOUT_WORLD);
};

void stokesSolver::pointObsMatrix(double obs[][4], int size){
	MatCreate(MPI_COMM_SELF, &state2obsMatrix);
	MatSetSizes(state2obsMatrix, PETSC_DECIDE, PETSC_DECIDE, 2*size, 2*num_node_Q1isoQ2+num_node_Q1);
	MatSetFromOptions(state2obsMatrix);
	MatSeqAIJSetPreallocation(state2obsMatrix, 6, NULL);
	MatMPIAIJSetPreallocation(state2obsMatrix, 6, NULL, 6, NULL);	

	double element_h = 1.0/division_Q1isoQ2;

	int column;
	int row;

	double local_x;
	double local_y;

	int square[4];
	double coords[4];

	double epsilon;
	double eta;

	for (int i = 0; i < size; i++){
		column  = obs[i][0] / element_h;
		row     = obs[i][1] / element_h;
		local_x = fmod(obs[i][0], element_h);
		local_y = fmod(obs[i][1], element_h);

		square[0] = quadrilaterals[0][division_Q1isoQ2*row+column];
		square[1] = quadrilaterals[1][division_Q1isoQ2*row+column];
		square[2] = quadrilaterals[2][division_Q1isoQ2*row+column];
		square[3] = quadrilaterals[3][division_Q1isoQ2*row+column];

		coords[0] =	points[0][square[0]];
		coords[1] =	points[0][square[2]];
		coords[2] =	points[1][square[0]];
		coords[3] =	points[1][square[2]];		

		epsilon = (obs[i][0]-coords[0])/(coords[1]-coords[0])*2-1;
		eta     = (obs[i][1]-coords[2])/(coords[3]-coords[2])*2-1;

		MatSetValue(state2obsMatrix, i, square[0], 0.25*(1.0-epsilon)*(1.0-eta), INSERT_VALUES);
		MatSetValue(state2obsMatrix, i, square[1], 0.25*(1.0+epsilon)*(1.0-eta), INSERT_VALUES);
		MatSetValue(state2obsMatrix, i, square[2], 0.25*(1.0+epsilon)*(1.0+eta), INSERT_VALUES);
		MatSetValue(state2obsMatrix, i, square[3], 0.25*(1.0-epsilon)*(1.0+eta), INSERT_VALUES);

		MatSetValue(state2obsMatrix, i+size, num_node_Q1isoQ2+square[0], 0.25*(1.0-epsilon)*(1.0-eta), INSERT_VALUES);
		MatSetValue(state2obsMatrix, i+size, num_node_Q1isoQ2+square[1], 0.25*(1.0+epsilon)*(1.0-eta), INSERT_VALUES);
		MatSetValue(state2obsMatrix, i+size, num_node_Q1isoQ2+square[2], 0.25*(1.0+epsilon)*(1.0+eta), INSERT_VALUES);
		MatSetValue(state2obsMatrix, i+size, num_node_Q1isoQ2+square[3], 0.25*(1.0-epsilon)*(1.0+eta), INSERT_VALUES);
   	}
	MatAssemblyBegin(state2obsMatrix, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(state2obsMatrix, MAT_FINAL_ASSEMBLY);
	// MatView(state2obsMatrix, PETSC_VIEWER_STDOUT_WORLD);
}


void stokesSolver::adjointEqn(int size, double obs[][4]){
	Vec workspace;
 	VecCreate(MPI_COMM_SELF, &workspace);
	VecSetSizes(workspace, PETSC_DECIDE, 2*size);
	VecSetFromOptions(workspace);
	VecSet(workspace, 0.0);
	VecAssemblyBegin(workspace);
	VecAssemblyEnd(workspace);


 	MatMult(state2obsMatrix, states, workspace);
	Vec workspace2;
 	VecCreate(MPI_COMM_SELF, &workspace2);
	VecSetSizes(workspace2, PETSC_DECIDE, 2*size);
	VecSetFromOptions(workspace2);
 	for (int i = 0; i < size; ++i){
 		VecSetValue(workspace2, i, obs[i][2], INSERT_VALUES);
 		VecSetValue(workspace2, size+i, obs[i][3], INSERT_VALUES);
 	}
	VecAssemblyBegin(workspace2);
	VecAssemblyEnd(workspace2);
 	VecAXPY(workspace, -1, workspace2);
 	MatMultTranspose(state2obsMatrix, workspace, adjointRhs);

 	VecDuplicate(adjointRhs, &adjoints);
 	apply_homo_boundary_condition();
	KSPSolve(ksp, adjointRhs, adjoints);	
}

void stokesSolver::controlEqn(double grad[]){
	VecGetArray(adjoints, &adjoint_array);
	double workspace;
	double load[2*num_node_Q1isoQ2];
	std::cout << "gradient: ";
	for (int i = 0; i < 10; ++i){
		load_vector_i(i, load);
		workspace = 0;
		for (int j = 0; j < 2*num_node_Q1isoQ2; ++j){
			workspace += load[i]*adjoint_array[i];
		}
		grad[i] = u[i] - workspace;	
		std::cout << grad[i] << " ";
	}
	std::cout << std::endl;
}

void stokesSolver::WuuApply(Vec direction, int size){
	Vec workspace;
 	VecCreate(MPI_COMM_SELF, &workspace);
	VecSetSizes(workspace, PETSC_DECIDE, 2*size);
	VecSetFromOptions(workspace);
	VecSet(workspace, 0.0);
	VecAssemblyBegin(workspace);
	VecAssemblyEnd(workspace);

	MatMult(state2obsMatrix, direction, workspace);
	MatMultTranspose(state2obsMatrix, workspace, direction);
}

Vec stokesSolver::CApply(Vec direction, int size, bool T){
	Mat C;
	MatCreate(MPI_COMM_SELF, &C);
	MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, 2*num_node_Q1isoQ2+num_node_Q1, 10);
	MatSetFromOptions(C);
	MatSeqAIJSetPreallocation(C, 10, NULL);
	MatMPIAIJSetPreallocation(C, 10, NULL, 10, NULL);	
	double load[2*num_node_Q1isoQ2];
	for (int i = 0; i < 10; ++i){
		load_vector_i(i, load);
		for (int j = 0; j < 2*num_node_Q1isoQ2+num_node_Q1; ++j){
			MatSetValue(C, j, i, load[j], INSERT_VALUES);
		}
	}
	MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);

	Vec workspace;
	if (T == 0){
		VecSetSizes(workspace, PETSC_DECIDE, 2*num_node_Q1isoQ2+num_node_Q1);
		VecSet(workspace, 0.0);
		VecAssemblyBegin(workspace);
		VecAssemblyEnd(workspace);
		MatMult(C, direction, workspace);
	} else {
		VecSetSizes(workspace, PETSC_DECIDE, 20);
		VecSet(workspace, 0.0);
		VecAssemblyBegin(workspace);
		VecAssemblyEnd(workspace);
		MatMultTranspose(C, direction, workspace);
	}
	return workspace;
}

Vec stokesSolver::ASolve(Vec direction, int size, bool T){
	Vec workspace;
	if (T == 0){
		KSPSolve(ksp, direction, workspace);
	} else {
		KSPSolveTranspose(ksp, direction, workspace);
	}
	return workspace;
}

Vec stokesSolver::RApply(Vec direction, bool T){
	Mat R;
	MatCreate(MPI_COMM_SELF, &R);
	MatSetSizes(R, PETSC_DECIDE, PETSC_DECIDE, 10, 10);
	MatSetFromOptions(R);
	MatSeqAIJSetPreallocation(R, 1, NULL);
	MatMPIAIJSetPreallocation(R, 1, NULL, 1, NULL);	
	for (int i = 0; i < 10; ++i){
		MatSetValue(R, i, i, 1.0, INSERT_VALUES);
	}
	MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY);

	Vec workspace;
	VecSetSizes(workspace, PETSC_DECIDE, 10);
	VecSet(workspace, 0.0);
	VecAssemblyBegin(workspace);
	VecAssemblyEnd(workspace);
	if (T == 0){
		MatMult(R, direction, workspace);
	} else {
		MatMultTranspose(R, direction, workspace);
	}

	return workspace;
}

void stokesSolver::HessianAction(double grad[]){
	Vec workspace;
	VecSetSizes(workspace, PETSC_DECIDE, 10);
	for (int i = 0; i < 10; ++i){
		VecSetValue(workspace, i, grad[i], INSERT_VALUES);
	}
	VecAssemblyBegin(workspace);
	VecAssemblyEnd(workspace);	

	Vec workspace2;
	workspace2 = RApply(workspace, false);

	Vec workspace3;
	Vec workspace4;
	workspace3 = CApply(workspace, 10, false);
	workspace4 = ASolve(workspace3, 10, false);
	WuuApply(workspace4, 10);
	workspace3 = ASolve(workspace4, 10, true);
	workspace = CApply(workspace4, 10, true);

	VecAXPY(workspace, 1, workspace4);
	VecGetArray(workspace, &grad);
}

void stokesSolver::fullHessian(){
	double hessian[num_term][num_term];
	double input[num_term];
	for (int i = 0; i < num_term; ++i){
		for (int j = 0; j < num_term; ++j){
			if (i == j){
				input[j] = 1;
			} else {
				input[j] = 0;
			}
		}
		HessianAction(input);
		for (int j = 0; j< num_term; ++j){
			hessian[j][i] = input[j];
			std::cout << input[j] << " ";
		}
		std::cout << std::endl;
	}
}

// void stokesSolver::ReducedHessian(){
// 	// std::cout << "create random matrix" << std::endl;
// 	double randomOmega[num_term][15];
// 	for (unsigned i = 0; i < num_term; i++){
// 		for (unsigned j = 0; j < 15; j++){
// 			randomOmega[i][j] = distribution(generator);
// 		}
// 	}	

// 	double inputSpace[num_term];
// 	double outputSpace[num_term];
// 	MatCreateSeqDense(geo->comm, num_term, 15, NULL, &yMat);
// 	for (int i = 0; i < 15; i++){
// 		for (int j = 0; j < num_term; ++j){
// 			inputSpace[j] = randomOmega[j][i];
// 		}
// 		HessianAction(inputSpace);

// 		for (int j = 0; j < num_term; j++){
// 			MatSetValue(yMat, j, i, inputSpace[j], INSERT_VALUES);
// 		}
// 	}
// 	MatAssemblyBegin(yMat, MAT_FINAL_ASSEMBLY);
// 	MatAssemblyEnd(yMat, MAT_FINAL_ASSEMBLY);

// 	// std::cout << "yMat size" << std::endl;
// 	// MatView(yMat, PETSC_VIEWER_STDOUT_WORLD);

// 	// std::cout << "QR decomposition" << std::endl;

// 	BV bv;
// 	Mat Q;
// 	BVCreateFromMat(yMat, &bv);
// 	BVSetFromOptions(bv);
// 	BVOrthogonalize(bv, NULL);
// 	BVCreateMat(bv, &Q);

// 	// std::cout << "initialized eigenvalue problem" << std::endl;

// 	EPS eps;
// 	Mat T, S, lambda, U, UTranspose, ReducedHessian;
// 	Vec workSpace1, workspace2, xr, xi;
// 	PetscScalar kr, ki;
// 	MatCreate(geo->comm, &T);
// 	MatCreate(geo->comm, &S);
// 	MatCreate(geo->comm, &U);
// 	MatCreate(geo->comm, &lambda);
// 	VecCreate(geo->comm, &workSpace1);
// 	MatSetSizes(T, PETSC_DECIDE, PETSC_DECIDE, 15, 15);
// 	MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, 15, 15);
// 	MatSetSizes(U, PETSC_DECIDE, PETSC_DECIDE, 15, 15);
// 	MatSetSizes(lambda, PETSC_DECIDE, PETSC_DECIDE, 15, 15);
// 	VecSetSizes(workSpace1, PETSC_DECIDE, 15);
// 	MatSetFromOptions(T);
// 	MatSetFromOptions(S);
// 	MatSetFromOptions(U);
// 	MatSetFromOptions(lambda);
// 	VecSetFromOptions(workSpace1);
// 	MatSetUp(T);
// 	MatSetUp(S);
// 	MatSetUp(lambda);
// 	MatSetUp(U);
// 	VecSetUp(workSpace1);

// 	// std::cout << "Assemble T matrix" << std::endl;
// 	for (int i = 0; i < 15; i++){
// 		MatGetColumnVector(Q, workspace1, i);
// 		VecGetArray(workspace1, &inputSpace);

// 		HessianAction(inputSpace);
// 		for (int j = 0; j < num_term; ++j){
// 			VecSetValue(workspace1, j, inputSpace[j]);		
// 		}
// 		VecAssemblyBegin(workspace1);
// 		VecAssemblyEnd(workspace1);

// 		MatMultTranspose(Q, workspace1, workSpace2);
// 		VecGetArray(workSpace2, &outputSpace);
// 		for (int j = 0; j < 15; j++){
// 			MatSetValue(T, j, i, outputSpace[j], INSERT_VALUES);
// 		}
// 	}
// 	MatAssemblyBegin(T, MAT_FINAL_ASSEMBLY);
// 	MatAssemblyEnd(T, MAT_FINAL_ASSEMBLY);
// 	MatCreateVecs(T, NULL, &xr);
// 	MatCreateVecs(T, NULL, &xi);

// 	// std::cout << "start EPS solver" << std::endl;

// 	EPSCreate(PETSC_COMM_SELF, &eps);
// 	EPSSetOperators(eps, T, NULL);
// 	EPSSetFromOptions(eps);
// 	EPSSolve(eps);

// 	PetscScalar *eigenValues;
// 	for (int i = 0; i < 15; i++){
// 		EPSGetEigenpair(eps, i, &kr, &ki, xr, xi);
// 		VecGetArray(xr, &eigenValues);
// 		for (int j = 0; j < 15; j++){
// 			if (i == j){
// 				MatSetValue(lambda, j, i, kr, INSERT_VALUES);
// 			} else {
// 				MatSetValue(lambda, j, i, 0, INSERT_VALUES);
// 			}
// 			MatSetValue(S, j, i, eigenValues[j], INSERT_VALUES);
// 		}
// 	}

// 	MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
// 	MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);
// 	MatAssemblyBegin(lambda, MAT_FINAL_ASSEMBLY);
// 	MatAssemblyEnd(lambda, MAT_FINAL_ASSEMBLY);

// 	std::cout << "compute hessian matrix" << std::endl;

// 	MatMatMult(Q, S, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &U);
// 	MatTranspose(U, MAT_INITIAL_MATRIX, &UTranspose);

// 	// MatPtAP(lambda, UTranspose, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &ReducedHessian);
// 	// MatView(ReducedHessian, PETSC_VIEWER_STDOUT_WORLD);

// 	// std::cout << "convert PETSc to Dolfin" << std::endl;
// 	// std::shared_ptr<Matrix> out(new Matrix(PETScMatrix(ReducedHessian)));

// 	// std::cout << "Mat Destroy Reduced Hessian" << std::endl;
// 	// MatDestroy(&ReducedHessian);

// 	// std::cout << "hessianMatrix" << std::endl;
// }

double stokesSolver::solve(double obs[][4], int size)
{
	forwardEqn();
	return lnlikelihood(obs, size);
}

void stokesSolver::pCNProposalGenerator(double proposal[])
{
	for (int m_idx = 0; m_idx < num_term; ++m_idx){
		proposal[m_idx] = sqrt(1-beta*beta)*proposal[m_idx] + beta*distribution(generator);
	}
}

void stokesSolver::SNProposalGenerator(double proposal[]){
	
}

double stokesSolver::lnlikelihood(double obs[][4], int size)
{
	double element_h = 1.0/division_Q1isoQ2;

	int column;
	int row;

	double local_x;
	double local_y;

	int square[4];
	double coords[4];

	double epsilon;
	double eta;

	double point_u1;
	double point_u2;

	double local_u1[4];
	double local_u2[4];

	double misfit = 0;

	for (int i = 0; i < size; i++){
		column  = obs[i][0] / element_h;
		row     = obs[i][1] / element_h;
		local_x = fmod(obs[i][0], element_h);
		local_y = fmod(obs[i][1], element_h);

		square[0] = quadrilaterals[0][division_Q1isoQ2*row+column];
		square[1] = quadrilaterals[1][division_Q1isoQ2*row+column];
		square[2] = quadrilaterals[2][division_Q1isoQ2*row+column];
		square[3] = quadrilaterals[3][division_Q1isoQ2*row+column];

		coords[0] =	points[0][square[0]];
		coords[1] =	points[0][square[2]];
		coords[2] =	points[1][square[0]];
		coords[3] =	points[1][square[2]];		

		epsilon = (obs[i][0]-coords[0])/(coords[1]-coords[0])*2-1;
		eta     = (obs[i][1]-coords[2])/(coords[3]-coords[2])*2-1;

		for (int k = 0; k < 4; ++k){
			local_u1[k] = u1[square[k]];
			local_u2[k] = u2[square[k]];
		}

		point_u1 = shape_interpolation(epsilon, eta, local_u1);
		point_u2 = shape_interpolation(epsilon, eta, local_u2);
      
      	//std::cout << "point u: " << point_u1 << " " << point_u2 << " obs: " << obs[i][2] << " " << obs[i][3] << std::endl; 
        misfit += std::pow(point_u1-obs[i][2], 2.) + std::pow(point_u2-obs[i][3], 2.);
	}
	return -misfit/noiseVariance;
}

void obs_gen(int level, std::vector<double> &rand_coef, double obs[][4], int obsNumPerRow, double noiseVariance)
{
	stokesSolver obsSolver(level, rand_coef.size(), noiseVariance);
	for (int i = 0; i < rand_coef.size(); ++i){
		obsSolver.u[i] = rand_coef[i];
	}
	obsSolver.forwardEqn();

	int obsIdx[obsNumPerRow*obsNumPerRow];
	double incremental = (1./obsNumPerRow)/(1./obsSolver.division_Q1isoQ2);
    double Idx1 = (1./2./obsNumPerRow)/(1./obsSolver.division_Q1isoQ2) + (1./2./obsNumPerRow)/(1./obsSolver.division_Q1isoQ2)*(obsSolver.division_Q1isoQ2+1.);

    for (int i = 0; i < obsNumPerRow; i++){  
    	obsIdx[obsNumPerRow*i] = Idx1 + i*incremental*(obsSolver.division_Q1isoQ2+1);
    }

	for (int i = 0; i < obsNumPerRow; i++){
		for (int j = 0; j < obsNumPerRow; j++){
			obsIdx[obsNumPerRow*i+j] = obsIdx[obsNumPerRow*i]+j*incremental;   		
		}
	}    
 
	for (int i = 0; i < obsNumPerRow*obsNumPerRow; i++){
		obs[i][0] = obsSolver.points[0][obsIdx[i]]; 
		obs[i][1] = obsSolver.points[1][obsIdx[i]];
		obs[i][2] = obsSolver.u1[obsIdx[i]];
		obs[i][3] = obsSolver.u2[obsIdx[i]];
	}
}

void plain_mcmc(stokesSolver *solver, int max_samples, double obs[][4], int obs_size, double mean[], int coef_size)
{
    std::uniform_real_distribution<double> uni_dist(0.0, 1.0);
    double u[coef_size];
    double u_next[coef_size];
    double sum[coef_size];
    double uniform_alpha;
    double alpha;

    for (int i = 0; i < coef_size; ++i){
    	solver->u[i] = solver->distribution(solver->generator);
    	u[i] = solver->u[i];
    	u_next[i] = solver->u[i];
    	sum[i] = solver->u[i];
    }

    double lnlikelihood;
    double lnlikelihood_next;
    lnlikelihood = solver->solve(obs, obs_size);

    for (int i = 0; i < max_samples; i++){
        solver->pCNProposalGenerator(u_next);
        for (int m_idx = 0; m_idx < coef_size; ++m_idx){
            solver->u[m_idx] = u_next[m_idx];
        }        

	lnlikelihood_next = solver->solve(obs, obs_size);

        alpha = std::min(0.0, lnlikelihood_next - lnlikelihood);
        uniform_alpha = std::log(uni_dist(solver->generator));

        if (uniform_alpha < alpha){
            std::cout << "accepted: " << uniform_alpha << ", " << alpha << std::endl; 
            for (int m_idx = 0; m_idx < coef_size; ++m_idx){
                u[m_idx] = u_next[m_idx];
            }
            lnlikelihood = lnlikelihood_next;
        } else {         
	    std::cout << "rejected: " << uniform_alpha << ", " << alpha << std::endl; 
            for (int m_idx = 0; m_idx < coef_size; ++m_idx){
                u_next[m_idx] = u[m_idx];
            }	
        }
        for (int m_idx = 0; m_idx < coef_size; ++m_idx){
            sum[m_idx] += u[m_idx];
             std::cout << u[m_idx] << " ";
        }
        //std::cout << std::endl;
        // if (i % 10 == 0){
        // 	std::cout << "average: " << sum[0]/(i+1) << " " << sum[1]/(i+1) << " " << sum[2]/(i+2) << std::endl;
        // }
    }
    //std::cout << "final mean: " << std::endl;
    for (int m_idx = 0; m_idx < coef_size; ++m_idx){
        mean[m_idx] = sum[m_idx] / (max_samples+1);
        //std::cout << mean[m_idx] << " ";
    }
}

void ml_mcmc(int levels, std::vector<std::shared_ptr<stokesSolver>> solvers, double obs[][4], double out[], int size, int a, int coef_size)
{
    std::uniform_real_distribution<double> uni_dist(0.0, 1.0);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    // int M = pow(levels, a-2)*pow(2.0, 2.0*levels);
    int M = levels*pow(2.0, 2.0*levels);
    double mean[coef_size];
    plain_mcmc(solvers[0].get(), M, obs, size, mean, coef_size);
    double L[levels+1][coef_size];
    std::cout << "L0: ";
    for (int m_idx = 0; m_idx < coef_size; ++m_idx){
        L[0][m_idx] = mean[m_idx];
        std::cout << mean[m_idx] << " ";
    }
    std::cout << std::endl;

    double u_upper[coef_size];
    double u_lower[coef_size];
    double u_next[coef_size];
    double lnlikelihoodUpper;
    double lnlikelihoodLower;
    double lnlikelihoodUpper_next;
    double lnlikelihoodLower_next;
    double lnlikelihoodUpper_Lower;
    double lnlikelihoodLower_Upper;
    double alpha;
    double uniform_alpha;
    int acceptance;
    int I;

    double A0[coef_size];
    double A1[coef_size];
    double A2[coef_size];
    double A3[coef_size];
    double A4[coef_size];
    double A5[coef_size];
    double A6[coef_size];
    double A7[coef_size];

    double A0Sum[coef_size];
    double A1Sum[coef_size];
    double A2Sum[coef_size];
    double A3Sum[coef_size];
    double A4Sum[coef_size];
    double A5Sum[coef_size];
    double A6Sum[coef_size];
    double A7Sum[coef_size];

    for (int i = 0; i < levels; i++){
        M = pow(i+1, 3)*pow(2.0,2.0*(levels-i-1.0));
	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
            A0Sum[m_idx] = 0;
            A1Sum[m_idx] = 0;
            A2Sum[m_idx] = 0;
            A3Sum[m_idx] = 0;
            A4Sum[m_idx] = 0;
            A5Sum[m_idx] = 0;
            A6Sum[m_idx] = 0;
            A7Sum[m_idx] = 0;
        }

/////////////////////////////////////////////////////////////////////

        std::shared_ptr<stokesSolver> solverUpper = solvers[i+1];
        std::shared_ptr<stokesSolver> solverLower = solvers[i];
        
        for (int m_idx = 0; m_idx < coef_size; ++m_idx){
            u_upper[m_idx] = norm_dist(solverLower->generator);
            u_lower[m_idx] = u_upper[m_idx];
            u_next[m_idx] = u_upper[m_idx];
            solverUpper->u[m_idx] = u_upper[m_idx];
            solverLower->u[m_idx] = u_lower[m_idx];
        }


        lnlikelihoodUpper = solverUpper->solve(obs, size);
        lnlikelihoodLower = solverLower->solve(obs, size);
        if (-lnlikelihoodUpper+lnlikelihoodLower < 0){
            I = 1;
        } else {
            I = 0;   
        }

        if (I == 0){
            for (int m_idx = 0; m_idx < coef_size; ++m_idx){
                A0[m_idx] = 0;
                A2[m_idx] = 0;
                A5[m_idx] = 0;
                A6[m_idx] = u_upper[m_idx]*(1-I);
                A0Sum[m_idx] += A0[m_idx];
                A2Sum[m_idx] += A2[m_idx];
                A5Sum[m_idx] += A5[m_idx];
                A6Sum[m_idx] += A6[m_idx];
            }
        } else {
            for (int m_idx = 0; m_idx < coef_size; ++m_idx){
                A0[m_idx] = (1-exp(-lnlikelihoodUpper+lnlikelihoodLower))*u_upper[m_idx]*I;
                A2[m_idx] = (exp(-lnlikelihoodUpper+lnlikelihoodLower)-1)*I;
                A5[m_idx] = exp(-lnlikelihoodUpper+lnlikelihoodLower)*u_upper[m_idx]*I;
                A6[m_idx] = 0;
                A0Sum[m_idx] += A0[m_idx]; 
                A2Sum[m_idx] += A2[m_idx]; 
                A5Sum[m_idx] += A5[m_idx]; 
                A6Sum[m_idx] += A6[m_idx]; 
    	    }
        }


        if (I == 1){
            for (int m_idx = 0; m_idx < coef_size; ++m_idx){
                A1[m_idx] = 0; 
                A3[m_idx] = u_lower[m_idx]; 
                A4[m_idx] = 0; 
                A7[m_idx] = 0; 
                A1Sum[m_idx] += A1[m_idx]; 
                A3Sum[m_idx] += A3[m_idx]; 
                A4Sum[m_idx] += A4[m_idx]; 
                A7Sum[m_idx] += A7[m_idx];
            }                  
        } else {
            for (int m_idx = 0; m_idx < coef_size; ++m_idx){
                A1[m_idx] = (exp(-lnlikelihoodLower+lnlikelihoodUpper)-1)*u_lower[m_idx]*(1-I);
                A3[m_idx] = 0;
                A4[m_idx] = (1-exp(-lnlikelihoodLower+lnlikelihoodUpper))*(1-I);
                A7[m_idx] = exp(-lnlikelihoodLower+lnlikelihoodUpper)*u_lower[m_idx]*(1-I);
                A1Sum[m_idx] += A1[m_idx];
                A3Sum[m_idx] += A3[m_idx];
                A4Sum[m_idx] += A4[m_idx];
                A7Sum[m_idx] += A7[m_idx];     
    	    }
         }

        for (int sampleNum = 0; sampleNum < M; sampleNum++){
            solverUpper->pCNProposalGenerator(u_next);
            for (int m_idx = 0; m_idx < coef_size; ++m_idx){
                solverUpper->u[m_idx] = u_next[m_idx];
                solverLower->u[m_idx] = u_next[m_idx];   
            }

	    lnlikelihoodUpper_next = solverUpper->solve(obs, size);
	    lnlikelihoodLower_next = solverLower->solve(obs, size);


            alpha = std::min(0.0, lnlikelihoodUpper_next - lnlikelihoodUpper);
            uniform_alpha = log(uni_dist(solverUpper->generator));
            if (uniform_alpha < alpha){
                acceptance = 1;
            } else {
                acceptance = 0;
            }


            if (acceptance == 1){
            	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
            		u_upper[m_idx] = u_next[m_idx];
            	}

                lnlikelihoodUpper = lnlikelihoodUpper_next;
                lnlikelihoodUpper_Lower = lnlikelihoodLower_next;

                if (-lnlikelihoodUpper+lnlikelihoodUpper_Lower < 0){
                    I = 1;
                } else {
                    I = 0; 
                }

                if (I == 0){
                	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
                            A0[m_idx] = 0;
                            A2[m_idx] = 0;
                            A5[m_idx] = 0;
                            A6[m_idx] = u_upper[m_idx]*(1-I);
	                    A0Sum[m_idx] += A0[m_idx]; 
	                    A2Sum[m_idx] += A2[m_idx]; 
	                    A5Sum[m_idx] += A5[m_idx]; 
	                    A6Sum[m_idx] += A6[m_idx];  

                	}                          
                } else {
                	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
            		    A0[m_idx] = (1-exp(-lnlikelihoodUpper+lnlikelihoodUpper_Lower))*u_upper[m_idx]*I;
               	            A2[m_idx] = (exp(-lnlikelihoodUpper+lnlikelihoodUpper_Lower)-1)*I;
                	    A5[m_idx] = exp(-lnlikelihoodUpper+lnlikelihoodUpper_Lower)*u_upper[m_idx]*I;
                	    A6[m_idx] = 0; 
	                    A0Sum[m_idx] += A0[m_idx];
	                    A2Sum[m_idx] += A2[m_idx];
	                    A5Sum[m_idx] += A5[m_idx];
	                    A6Sum[m_idx] += A6[m_idx];
                	}
            	}
                
            } else {
            	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
                    A0Sum[m_idx] += A0[m_idx]; 
                    A2Sum[m_idx] += A2[m_idx]; 
                    A5Sum[m_idx] += A5[m_idx]; 
                    A6Sum[m_idx] += A6[m_idx];    
            	}
	}

        alpha = std::min(0.0, lnlikelihoodLower_next - lnlikelihoodLower);
        uniform_alpha = log(uni_dist(solverLower->generator));
        if (uniform_alpha < alpha){
            acceptance = 1;
        } else {
            acceptance = 0;
        }


        if (acceptance == 1){
            for (int m_idx = 0; m_idx < coef_size; ++m_idx){
                u_lower[m_idx] = u_next[m_idx];
            }

            lnlikelihoodLower = lnlikelihoodLower_next;
            lnlikelihoodLower_Upper = lnlikelihoodUpper_next;

            if (-lnlikelihoodLower_Upper+lnlikelihoodLower < 0){
                I = 1;
            } else {
                I = 0; 
            }

            if (I == 1){
            	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
	                A1[m_idx] = 0; 
	                A3[m_idx] = u_lower[m_idx]; 
	                A4[m_idx] = 0; 
	                A7[m_idx] = 0; 
	                A1Sum[m_idx] += A1[m_idx]; 
	                A3Sum[m_idx] += A3[m_idx]; 
	                A4Sum[m_idx] += A4[m_idx]; 
	                A7Sum[m_idx] += A7[m_idx];   
            	}        
            } else {
            	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
	                A1[m_idx] = (exp(-lnlikelihoodLower+lnlikelihoodLower_Upper)-1)*u_lower[m_idx]*(1-I);
	                A3[m_idx] = 0;
	                A4[m_idx] = (1-exp(-lnlikelihoodLower+lnlikelihoodLower_Upper))*(1-I);
	                A7[m_idx] = exp(-lnlikelihoodLower+lnlikelihoodLower_Upper)*u_lower[m_idx]*(1-I);
	                A1Sum[m_idx] += A1[m_idx];
	                A3Sum[m_idx] += A3[m_idx];
	                A4Sum[m_idx] += A4[m_idx];
	                A7Sum[m_idx] += A7[m_idx];  
            	}
        		}
            } else {
            	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
                    A1Sum[m_idx] += A1[m_idx];
                    A3Sum[m_idx] += A3[m_idx];
                    A4Sum[m_idx] += A4[m_idx];
                    A7Sum[m_idx] += A7[m_idx];  
            	}
	    }
	}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

    	std::cout << "L" << i+1 << ": ";
    	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
	        L[i+1][m_idx] = A0Sum[m_idx]/(M+1)+A1Sum[m_idx]/(M+1)+A2Sum[m_idx]/(M+1)*(A3Sum[m_idx]/(M+1)+A7Sum[m_idx]/(M+1))+A4Sum[m_idx]/(M+1)*(A5Sum[m_idx]/(M+1)+A6Sum[m_idx]/(M+1));
    		std::cout << L[i+1][m_idx] << " " << A0Sum[m_idx] << " " << A1Sum[m_idx]<< " " << A2Sum[m_idx]<< " " << A3Sum[m_idx]<< " " << A4Sum[m_idx]<< " " << A5Sum[m_idx]<< " " << A6Sum[m_idx]<< " " << A7Sum[m_idx] << std::endl;   
	}
    	std::cout << std::endl;
    }
    for (int m_idx = 0; m_idx < coef_size; ++m_idx){
    	out[m_idx] = 0.0;
    }
    for (int i = 0; i < levels; i++){
	    for (int m_idx = 0; m_idx < coef_size; ++m_idx){
	    	out[m_idx] += L[i][m_idx];
	    }
    }
}

void write2txt(double array[][4], int arraySize, std::string pathName){
	std::ofstream myfile;
	myfile.open(pathName);
	for (int j = 0; j < 4; ++j){
		for (int i = 0; i < arraySize; ++i){
			myfile << array[i][j] << " ";
		}
		myfile << std::endl;
	}
	myfile.close();
};


void txt2read(double array[][4], int arraySize, std::string pathName){
	std::ifstream myfile;
	myfile.open(pathName, std::ios_base::in);
	for (int j = 0; j < 4; ++j){
		for(int i = 0; i < arraySize; ++i){
			myfile >> array[i][j];
		}
	}
	myfile.close();
}

void read_config(std::vector<std::string> &paras, std::vector<std::string> &vals){
	std::ifstream cFile("mlmcmc_config.txt");
	if (cFile.is_open()){
		std::string line;
		while(getline(cFile, line)){
			line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
			if (line[0] == '#' || line.empty()){
				continue;
			}
			auto delimiterPos = line.find("=");
			auto name = line.substr(0, delimiterPos);
			auto value = line.substr(delimiterPos+1);

			paras.push_back(name);
			vals.push_back(value);
		}
	}
	cFile.close();
}

int main(int argc, char* argv[]){
	int rank;
	int size;

	PetscInitialize(&argc, &argv, NULL, NULL);
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	std::vector<std::string> name;
	std::vector<std::string> value;
	std::vector<double> rand_coef;
	read_config(name, value);
	int num_term = std::stoi(value[0]);
	for (int i = 0; i < num_term; ++i){
		rand_coef.push_back(std::stod(value[i+1]));
	}
	int levels = std::stoi(value[num_term+1]);
	int a = std::stoi(value[num_term+2]);
	double pCN = std::stod(value[num_term+3]);
	int task = std::stoi(value[num_term+4]);
	int plainMCMC_sample_number = std::stoi(value[num_term+5]);
	int obsNumPerRow = std::stoi(value[num_term+6]);
	double noiseVariance = std::stod(value[num_term+7]);

	if (rank == 0){
		std::cout << "configuration: " << std::endl; 
		std::cout << "num_term: " << num_term << " coefs: ";
		for (int i = 0; i < num_term; ++i){
			std::cout << rand_coef[i] << " ";
		}
		std::cout << std::endl;
		std::cout << "levels: "       << levels       <<  std::endl;
		std::cout << "a: "            << a            <<  std::endl;
		std::cout << "pCN: "          << pCN          <<  std::endl;
		std::cout << "task: "         << task         <<  std::endl;
		std::cout << "plainMCMC samples:"<< plainMCMC_sample_number << std::endl;
		std::cout << "obsNumPerRow: " << obsNumPerRow <<  std::endl;
		std::cout << "noiseVariance: " << noiseVariance << std::endl; 
	}
	//Generation of Observation
	double obs[obsNumPerRow*obsNumPerRow][4];
	std::string obs_file = std::to_string(levels+1);
	obs_file.append("_obs_para_");
	obs_file.append(std::to_string(num_term));
	obs_file.append("_");
	obs_file.append(std::to_string(obsNumPerRow));
	obs_file.append("_wNoise_");
	obs_file.append(std::to_string(noiseVariance));

	if (task == 0){ //Generate Observation
		obs_gen(levels+1, rand_coef, obs, obsNumPerRow, noiseVariance);

		std::normal_distribution<double> distribution{0.0, sqrt(noiseVariance)};
		std::default_random_engine generator;
		for (int i = 0; i < obsNumPerRow*obsNumPerRow; ++i){
			obs[i][2] = obs[i][2] + distribution(generator);
			obs[i][3] = obs[i][3] + distribution(generator);
		}
		write2txt(obs, obsNumPerRow*obsNumPerRow, obs_file);
	}

	if (task == 1){ //Run plain MCMC
		txt2read(obs, obsNumPerRow*obsNumPerRow, obs_file);
		stokesSolver* samplerSolver = new stokesSolver(levels, num_term, noiseVariance);
		samplerSolver->updateGeneratorSeed(rank*13.0);
		double mean[num_term];
		plain_mcmc(samplerSolver, plainMCMC_sample_number, obs, obsNumPerRow*obsNumPerRow, mean, num_term);
		delete samplerSolver;

		std::string outputfile = "plain_mcmc_output_";
		outputfile.append(std::to_string(rank));

		std::ofstream myfile;
		myfile.open(outputfile);
		for (int i = 0; i<num_term; ++i){
			myfile << mean[i] << " ";
		}
		myfile << std::endl;
		myfile.close();

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0){
			double buffer;
			std::string finaloutput = "plainmcmc_finalOutput";
			std::ofstream outputfile;
			outputfile.open(finaloutput);
			for (int i = 0; i < size; i++){
				std::string finalinput = "plain_mcmc_output_";
				finalinput.append(std::to_string(i));
				std::ifstream inputfile;
				inputfile.open(finalinput, std::ios_base::in);
				for(int i = 0; i < num_term; ++i){
					inputfile >> buffer;
					outputfile << buffer << " ";
				}
				outputfile << std::endl;
				inputfile.close();
			}
			outputfile.close();
		}
	}

	if (task == 2){ //Run MLMCMC
		txt2read(obs, obsNumPerRow*obsNumPerRow, obs_file);
		std::vector<std::shared_ptr<stokesSolver>> solvers(levels+1);
		for (int i = 0; i < levels+1; i++){
			solvers[i] = std::make_shared<stokesSolver>(i, num_term, noiseVariance);
			solvers[i]->beta = pCN;
			solvers[i]->updateGeneratorSeed(rank*13.0);
		} 
		double out[num_term];
		ml_mcmc(levels, solvers, obs, out, obsNumPerRow*obsNumPerRow, a, num_term);

		std::string outputfile = "output_";
		outputfile.append(std::to_string(rank));

		std::ofstream myfile;
		myfile.open(outputfile);
		for (int i = 0; i<num_term; ++i){
			myfile << out[i] << " ";
		}
		myfile << std::endl;
		myfile.close();

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0){
			double buffer;
			std::string finaloutput = "finalOutput";
			std::ofstream outputfile;
			outputfile.open(finaloutput);
			for (int i = 0; i < size; i++){
				std::string finalinput = "output_";
				finalinput.append(std::to_string(i));
				std::ifstream inputfile;
				inputfile.open(finalinput, std::ios_base::in);
				for(int i = 0; i < num_term; ++i){
					inputfile >> buffer;
					outputfile << buffer << " ";
				}
				outputfile << std::endl;
				inputfile.close();
			}
			outputfile.close();
		}
	}

	if (task == 3){ //Solver test
		stokesSolver testSolver(levels, num_term, noiseVariance);
		testSolver.forwardEqn();
	}

	PetscFinalize();
	return 0;
}
