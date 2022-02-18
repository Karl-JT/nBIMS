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

// #include <MLMCMC_Bi.h>
// #include <MLMCMC_Bi_Uniform.h>
#include <linearAlgebra.h>

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

	Mat M;
	Mat A;
	Mat C;

	Mat B1;
	Mat B2;	
	// Mat B1Cond;
	// Mat B2Cond;
	// Mat D;

	KSP kspm;
	KSP kspa;
	PC  pcm;
	PC  pca;

public:
	Vec statesx;
	Vec statesy;
	Vec statesp;
	Vec V;

	Mat s1SysMatrix;
	Mat s2SysMatrix;

	int division0 = 2;
	int division_Q1isoQ2;
	int division_Q1;
	int num_Q1isoQ2_element;
	int num_Q1_element;
	int num_node_Q1isoQ2;
	int num_node_Q1;

	std::unique_ptr<std::unique_ptr<double[]>[]> points;

	int level;
	int timeSteps;
	int num_term;
	double deltaT;
	double obs=-0.994214;
	double noiseVariance;
	double beta = 1;
	PetscScalar *states_array;
	std::unique_ptr<double[]> samples;
	std::unique_ptr<double[]> u1;
	std::unique_ptr<double[]> u2;

	std::default_random_engine generator;
	std::normal_distribution<double> normalDistribution{0.0, 1.0};
    // std::uniform_real_distribution<double> uniformDistribution{-1.0, 1.0};

	stokesSolver(int level_, int num_term_, double noiseVariance_);
	~stokesSolver(){
		MatDestroy(&A);
		MatDestroy(&B1);
		MatDestroy(&B2);
		// MatDestroy(&B1Cond);
		// MatDestroy(&B2Cond);
		MatDestroy(&C);
		// MatDestroy(&D);
		MatDestroy(&s1SysMatrix);
		MatDestroy(&s2SysMatrix);		
		VecDestroy(&statesx);	
		VecDestroy(&statesy);
		VecDestroy(&statesp);	
		KSPDestroy(&kspm);
		KSPDestroy(&kspa);
	};

	void createAllocations();
	double fsource(double x, double y, int dir);
	void shape_function(double epsilon, double eta, double N[]);
	double shape_interpolation(double epsilon, double eta, double x[4]);
	void jacobian_matrix(double x1[4], double x2[4], double epsilon, double eta, double J[2][2]);
	double jacobian_det(double J[2][2]);
	void jacobian_inv(double x1[4], double x2[4], double epsilon, double eta, double J[2][2]);
	void basis_function(double epsilon, double eta, double N[]);
	void basis_function_derivative(double dPhi[4][2], double epsilon, double eta);
	void hat_function_derivative(double dPhi[4][2], double epsilon, double eta, double x1[4], double x2[4]);
	void mass_matrix_element(double P[][4], double A[][4]);
	void M_matrix();
	void stiffness_matrix_element(double P[][4], double A[][4]);
	void A_matrix();
	void b1_matrix_element(double x1[4], double x2[4], double b1[4][4]);
	void b2_matrix_element(double x1[4], double x2[4], double b1[4][4]);
	void c1_matrix_element(double x1[4], double x2[4], double ux[4], double b1[4][4]);
	void c2_matrix_element(double x1[4], double x2[4], double uy[4], double b1[4][4]);
	void B1_matrix();
	void B2_matrix();
	void C_matrix();
	void D_matrix();
	void stage_one_matrix();
	void stage_two_matrix();
	void updateCmatrix();
	void load_vector_element(double x1[], double x2[], double v[], int dir);
	void load_vector(int dir);
	void apply_boundary_condition(Vec RhsVector);
	void linear_system_setup();
	void forwardEqn();
	void updateGeneratorSeed(double seed_);
	void getValues(double points[][2], double u[][2], int size);
	void solve();
};

stokesSolver::stokesSolver(int level_, int num_term_, double noiseVariance_) : level(level_), num_term(num_term_), noiseVariance(noiseVariance_) {
	samples = std::make_unique<double[]>(num_term_);
	timeSteps = std::pow(2, level_+1);
	deltaT = 0.1/timeSteps;

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
	Bs = std::make_unique<int[]>(size);
	int b_pos[4] = {0, division_Q1isoQ2, num_node_Q1isoQ2-1, num_node_Q1isoQ2-1-division_Q1isoQ2};	 
	for (int i=0; i<division_Q1isoQ2; i++){
		Bs[i]                    = b_pos[0]+i;
		Bs[i+division_Q1isoQ2]   = b_pos[1]+(division_Q1isoQ2+1)*i;
		Bs[i+2*division_Q1isoQ2] = b_pos[2]-i;
		Bs[i+3*division_Q1isoQ2] = b_pos[3]-(division_Q1isoQ2+1)*i;
	}

	createAllocations();
	
	u1 = std::make_unique<double[]>(num_node_Q1isoQ2);
	u2 = std::make_unique<double[]>(num_node_Q1isoQ2);
	VecGetArray(statesx, &states_array);
	std::copy(states_array, states_array+num_node_Q1isoQ2, u1.get());
	VecRestoreArray(statesx, &states_array);
	VecGetArray(statesy, &states_array);
	std::copy(states_array, states_array+num_node_Q1isoQ2, u2.get());

	linear_system_setup();

	KSPCreate(PETSC_COMM_SELF, &kspm);
	KSPSetType(kspm, KSPGMRES);
	KSPSetOperators(kspm, M, M);
	KSPGetPC(kspm, &pcm);
	PCSetType(pcm, PCJACOBI);
	PCJacobiSetType(pcm, PC_JACOBI_ROWSUM);
	PCSetUp(pcm);
	KSPSetTolerances(kspm, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetUp(kspm);

	KSPCreate(PETSC_COMM_SELF, &kspa);
	KSPSetType(kspa, KSPGMRES);
	KSPSetOperators(kspa, s2SysMatrix, s2SysMatrix);
	KSPGetPC(kspa, &pca);
	PCSetType(pca, PCJACOBI);
	PCSetUp(pca);
	KSPSetTolerances(kspa, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetUp(kspa);
};

void stokesSolver::createAllocations(){
 	VecCreate(MPI_COMM_SELF, &statesx);
	VecSetSizes(statesx, PETSC_DECIDE, num_node_Q1isoQ2);
	if (level > 3){
		VecSetType(statesx, VECSEQCUDA);
	} else {
		VecSetType(statesx, VECSEQ);
	}
	VecSet(statesx, 0.0);
	VecAssemblyBegin(statesx);
	VecAssemblyEnd(statesx);

 	VecCreate(MPI_COMM_SELF, &statesy);
	VecSetSizes(statesy, PETSC_DECIDE, num_node_Q1isoQ2);
	if (level > 3){
		VecSetType(statesy, VECSEQCUDA);
	} else {
		VecSetType(statesy, VECSEQ);
	}
	VecSet(statesy, 0.0);
	VecAssemblyBegin(statesy);
	VecAssemblyEnd(statesy);

 	VecCreate(MPI_COMM_SELF, &statesp);
	VecSetSizes(statesp, PETSC_DECIDE, num_node_Q1isoQ2);
	if (level > 3){
		VecSetType(statesp, VECSEQCUDA);
	} else {
		VecSetType(statesp, VECSEQ);
	}
	VecSet(statesp, 0.0);
	VecAssemblyBegin(statesp);
	VecAssemblyEnd(statesp);

 	VecCreate(MPI_COMM_SELF, &V);
	VecSetSizes(V, PETSC_DECIDE, num_node_Q1isoQ2);
	if (level > 3){
		VecSetType(V, VECSEQCUDA);
	} else {
		VecSetType(V, VECSEQ);
	}
	// VecSetFromOptions(V);

	MatCreate(PETSC_COMM_SELF, &M);
	MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, num_node_Q1isoQ2, num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(M, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(M, MATSEQAIJ);
	}
	MatSeqAIJSetPreallocation(M, 12, NULL);
	MatMPIAIJSetPreallocation(M, 12, NULL, 12, NULL);
	for (int i = 0; i < num_node_Q1isoQ2; ++i){
		MatSetValue(M, i, i, 0, INSERT_VALUES);
	}
	MatAssemblyBegin(M, MAT_FLUSH_ASSEMBLY);
	MatAssemblyEnd(M, MAT_FLUSH_ASSEMBLY);

	MatCreate(PETSC_COMM_SELF, &A);
	MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, num_node_Q1isoQ2, num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(A, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(A, MATSEQAIJ);
	}
	MatSeqAIJSetPreallocation(A, 12, NULL);
	MatMPIAIJSetPreallocation(A, 12, NULL, 12, NULL);
	for (int i = 0; i < num_node_Q1isoQ2; ++i){
		MatSetValue(A, i, i, 0, INSERT_VALUES);
	}
	MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

	MatCreate(PETSC_COMM_SELF, &B1);
	MatSetSizes(B1, PETSC_DECIDE, PETSC_DECIDE, num_node_Q1isoQ2, num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(B1, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(B1, MATSEQAIJ);
	}
	// MatSetFromOptions(B1);
	MatSeqAIJSetPreallocation(B1, 12, NULL);
	MatMPIAIJSetPreallocation(B1, 12, NULL, 12, NULL);


	MatCreate(MPI_COMM_SELF, &B2);
	MatSetSizes(B2, PETSC_DECIDE, PETSC_DECIDE, num_node_Q1isoQ2, num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(B2, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(B2, MATSEQAIJ);
	}
	// MatSetFromOptions(B2);
	MatSeqAIJSetPreallocation(B2, 12, NULL);
	MatMPIAIJSetPreallocation(B2, 12, NULL, 12, NULL);

	// MatCreate(MPI_COMM_SELF, &D);
	// MatSetSizes(D, PETSC_DECIDE, PETSC_DECIDE, num_node_Q1isoQ2, num_node_Q1);
	// if (level > 3){
	// 	MatSetType(D, MATSEQAIJCUSPARSE);
	// } else {
	// 	MatSetType(D, MATSEQAIJ);
	// }
	// // MatSetFromOptions(D);
	// MatSeqAIJSetPreallocation(D, 12, NULL);
	// MatMPIAIJSetPreallocation(D, 12, NULL, 12, NULL);
}

void stokesSolver::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};

double stokesSolver::fsource(double x, double y, int dir){
    double output;
    if (dir == 0){
	    output = samples[0]*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y); 
    } else {
    	output = -samples[0]*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y);
    }
    return output;
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



void stokesSolver::mass_matrix_element(double P[][4], double A[][4]){
	double N[4];
	double J[2][2];
	double Jdet[4];

	double refPoints[2];
	double refWeights[2];

	refPoints[0] = -1./sqrt(3.0);
	refPoints[1] =  1./sqrt(3.0);

	refWeights[0] = 1.0;
	refWeights[1] = 1.0;

	double gpPoints[4][4];
	double gpWeights[4];

	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 2; j++){
			basis_function(refPoints[i], refPoints[j], N);
			jacobian_matrix(P[0], P[1], refPoints[i], refPoints[j], J);
			gpWeights[2*i+j] = refWeights[i]*refWeights[j];
			Jdet[2*i+j] = jacobian_det(J);
			for (int k = 0; k < 4; ++k){
				gpPoints[2*i+j][k] = N[k];
			}
		}
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			A[i][j] = 0;
			for (int k = 0; k < 4; k++){
				A[i][j] += gpWeights[k]*gpPoints[k][i]*gpPoints[k][j]*Jdet[k];
			}
		}
	}
}

void stokesSolver::M_matrix(){
	double mass_element[4][4];
	double element_points[2][4];
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		mass_matrix_element(element_points, mass_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
 				MatSetValue(M, quadrilaterals[m][i], quadrilaterals[n][i], mass_element[m][n], ADD_VALUES);
			}
		}			
	}
	MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

void stokesSolver::stiffness_matrix_element(double P[][4], double A[][4]){
	double dPhi[4][2];
	double J[2][2];

	double refPoints[2];
	double refWeights[2];

	refPoints[0] = -1./sqrt(3.0);
	refPoints[1] =  1./sqrt(3.0);

	refWeights[0] = 1.0;
	refWeights[1] = 1.0;

	double gpPoints[4][8];
	double gpWeights[4];
	double Jdet[4];

	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 2; j++){
			hat_function_derivative(dPhi, refPoints[i], refPoints[j], P[0], P[1]);
			jacobian_matrix(P[0], P[1], refPoints[i], refPoints[j], J);
			gpWeights[2*i+j] = refWeights[i]*refWeights[j];
			Jdet[2*i+j] = jacobian_det(J);
			for (int k = 0; k < 8; ++k){
				gpPoints[2*i+j][k] = dPhi[k%4][k/4];
			}
		}
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			A[i][j] = 0;
			for (int k = 0; k < 4; k++){
				A[i][j] += gpWeights[k]*(gpPoints[k][i]*gpPoints[k][j]+gpPoints[k][4+i]*gpPoints[k][4+j])*Jdet[k];
			}
		}
	}
};

void stokesSolver::A_matrix(){
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
 				MatSetValue(A, quadrilaterals[m][i], quadrilaterals[n][i], stiffness_element[m][n], ADD_VALUES);
			}
		}			
	}
	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
};

void stokesSolver::b1_matrix_element(double x1[4], double x2[4], double b1[4][4]){
	double dPhi[4][2];
	double J[2][2];
	double N[4];

	double refPoints[2];
	double refWeights[2];

	refPoints[0] = -1./sqrt(3.0);
	refPoints[1] =  1./sqrt(3.0);

	refWeights[0] = 1.0;
	refWeights[1] = 1.0;

	double gpPoints1[4][4];
	double gpPoints2[4][4];
	double gpWeights[4];
	double Jdet[4];

	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 2; j++){
			basis_function(refPoints[i], refPoints[j], N);
			hat_function_derivative(dPhi, refPoints[i], refPoints[j], x1, x2);
			jacobian_matrix(x1, x2, refPoints[i], refPoints[j], J);
			gpWeights[2*i+j] = refWeights[i]*refWeights[j];
			Jdet[2*i+j] = jacobian_det(J);
			for (int k = 0; k < 4; ++k){
				gpPoints1[2*i+j][k] = dPhi[k][0];
				gpPoints2[2*i+j][k] = N[k];
			}
		}
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			b1[i][j] = 0;
			for (int k = 0; k < 4; k++){
				b1[i][j] += gpWeights[k]*(gpPoints1[k][j]*gpPoints2[k][i])*Jdet[k];
			}
		}
	}
};

void stokesSolver::b2_matrix_element(double x1[4], double x2[4], double b2[4][4]){
	double dPhi[4][2];
	double J[2][2];
	double N[4];

	double refPoints[2];
	double refWeights[2];

	refPoints[0] = -1./sqrt(3.0);
	refPoints[1] =  1./sqrt(3.0);

	refWeights[0] = 1.0;
	refWeights[1] = 1.0;

	double gpPoints1[4][4];
	double gpPoints2[4][4];
	double gpWeights[4];
	double Jdet[4];

	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 2; j++){
			basis_function(refPoints[i], refPoints[j], N);
			hat_function_derivative(dPhi, refPoints[i], refPoints[j], x1, x2);
			jacobian_matrix(x1, x2, refPoints[i], refPoints[j], J);
			gpWeights[2*i+j] = refWeights[i]*refWeights[j];
			Jdet[2*i+j] = jacobian_det(J);
			for (int k = 0; k < 4; ++k){
				gpPoints1[2*i+j][k] = dPhi[k][1];
				gpPoints2[2*i+j][k] = N[k];
			}
		}
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			b2[i][j] = 0;
			for (int k = 0; k < 4; k++){
				b2[i][j] += gpWeights[k]*(gpPoints1[k][j]*gpPoints2[k][i])*Jdet[k];
			}
		}
	}
};

void stokesSolver::c1_matrix_element(double x1[4], double x2[4], double ux[4], double c1[4][4]){
	double dPhi[4][2];
	double J[2][2];
	double N[4];

	double refPoints[2];
	double refWeights[2];

	refPoints[0] = -1./sqrt(3.0);
	refPoints[1] =  1./sqrt(3.0);

	refWeights[0] = 1.0;
	refWeights[1] = 1.0;

	double gpPoints0[4];
	double gpPoints1[4][4];
	double gpPoints2[4][4];
	double gpWeights[4];
	double Jdet[4];

	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 2; j++){
			basis_function(refPoints[i], refPoints[j], N);
			hat_function_derivative(dPhi, refPoints[i], refPoints[j], x1, x2);
			jacobian_matrix(x1, x2, refPoints[i], refPoints[j], J);
			gpWeights[2*i+j] = refWeights[i]*refWeights[j];
			Jdet[2*i+j] = jacobian_det(J);
			for (int k = 0; k < 4; ++k){
				gpPoints1[2*i+j][k] = dPhi[k][0];
				gpPoints2[2*i+j][k] = N[k];
			}
			gpPoints0[2*i+j] = N[0]*ux[0] + N[1]*ux[1] + N[2]*ux[2] + N[3]*ux[3];
		}
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			c1[i][j] = 0;
			for (int k = 0; k < 4; k++){
				c1[i][j] += gpWeights[k]*(gpPoints0[k]*gpPoints1[k][j]*gpPoints2[k][i])*Jdet[k];
			}
		}
	}
};

void stokesSolver::c2_matrix_element(double x1[4], double x2[4], double uy[4], double c2[4][4]){
	double dPhi[4][2];
	double J[2][2];
	double N[4];

	double refPoints[2];
	double refWeights[2];

	refPoints[0] = -1./sqrt(3.0);
	refPoints[1] =  1./sqrt(3.0);

	refWeights[0] = 1.0;
	refWeights[1] = 1.0;

	double gpPoints0[4];
	double gpPoints1[4][4];
	double gpPoints2[4][4];
	double gpWeights[4];
	double Jdet[4];

	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 2; j++){
			basis_function(refPoints[i], refPoints[j], N);
			hat_function_derivative(dPhi, refPoints[i], refPoints[j], x1, x2);
			jacobian_matrix(x1, x2, refPoints[i], refPoints[j], J);
			gpWeights[2*i+j] = refWeights[i]*refWeights[j];
			Jdet[2*i+j] = jacobian_det(J);
			for (int k = 0; k < 4; ++k){
				gpPoints1[2*i+j][k] = dPhi[k][1];
				gpPoints2[2*i+j][k] = N[k];
			}
			gpPoints0[2*i+j] = N[0]*uy[0] + N[1]*uy[1] + N[2]*uy[2] + N[3]*uy[3];
		}
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			c2[i][j] = 0;
			for (int k = 0; k < 4; k++){
				c2[i][j] += gpWeights[k]*(gpPoints0[k]*gpPoints1[k][j]*gpPoints2[k][i])*Jdet[k];
			}
		}
	}
};

void stokesSolver::B1_matrix(){
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
				MatSetValue(B1, quadrilaterals[n][i], quadrilaterals[m][i], b1_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(B1, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B1, MAT_FINAL_ASSEMBLY);	
	// MatTransposeMatMult(D, B1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &B1Cond);
};

void stokesSolver::B2_matrix(){
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
				MatSetValue(B2, quadrilaterals[n][i], quadrilaterals[m][i], b2_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(B2, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B2, MAT_FINAL_ASSEMBLY);		
};


void stokesSolver::C_matrix(){
	MatCreate(PETSC_COMM_SELF, &C);
	MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, num_node_Q1isoQ2, num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(C, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(C, MATSEQAIJ);
	}
	// MatSetFromOptions(A1);
	MatSeqAIJSetPreallocation(C, 24, NULL);
	MatMPIAIJSetPreallocation(C, 24, NULL, 24, NULL);

	double c1_element[4][4];
	double element_points[2][4];
	double ux[4];
	double uy[4];
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
			ux[j] = u1[quadrilaterals[j][i]];
		}
		c1_matrix_element(element_points[0], element_points[1], ux, c1_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
				MatSetValue(C, quadrilaterals[m][i], quadrilaterals[n][i], c1_element[n][m], ADD_VALUES);
			}
		}
	}

	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
			uy[j] = u2[quadrilaterals[j][i]];
		}
		c1_matrix_element(element_points[0], element_points[1], uy, c1_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
				MatSetValue(C, quadrilaterals[m][i], quadrilaterals[n][i], c1_element[n][m], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);	
};

// void stokesSolver::D_matrix(){
// 	for (int i = 0; i < num_node_Q1; i++){
// 		MatSetValue(D, i, i, 1, INSERT_VALUES);
// 	}
// 	for (int i = 0; i < num_Q1isoQ2_element; i++){
// 		if (Q1_idx[quadrilaterals[0][i]] < num_node_Q1isoQ2){
// 			MatSetValue(D, mesh_idx[quadrilaterals[1][i]], mesh_idx[quadrilaterals[0][i]], 0.5, INSERT_VALUES);
// 			MatSetValue(D, mesh_idx[quadrilaterals[3][i]], mesh_idx[quadrilaterals[0][i]], 0.5, INSERT_VALUES);
// 		} else if (Q1_idx[quadrilaterals[1][i]] < num_node_Q1isoQ2){
// 			MatSetValue(D, mesh_idx[quadrilaterals[2][i]], mesh_idx[quadrilaterals[1][i]], 0.5, INSERT_VALUES);
// 			MatSetValue(D, mesh_idx[quadrilaterals[0][i]], mesh_idx[quadrilaterals[1][i]], 0.5, INSERT_VALUES);			
// 		} else if (Q1_idx[quadrilaterals[2][i]] < num_node_Q1isoQ2){
// 			MatSetValue(D, mesh_idx[quadrilaterals[3][i]], mesh_idx[quadrilaterals[2][i]], 0.5, INSERT_VALUES);
// 			MatSetValue(D, mesh_idx[quadrilaterals[1][i]], mesh_idx[quadrilaterals[2][i]], 0.5, INSERT_VALUES);			
// 		} else if (Q1_idx[quadrilaterals[3][i]] < num_node_Q1isoQ2){
// 			MatSetValue(D, mesh_idx[quadrilaterals[0][i]], mesh_idx[quadrilaterals[3][i]], 0.5, INSERT_VALUES);
// 			MatSetValue(D, mesh_idx[quadrilaterals[2][i]], mesh_idx[quadrilaterals[3][i]], 0.5, INSERT_VALUES);			
// 		}
// 	}
// 	MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);
// 	MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);
// };

void stokesSolver::stage_one_matrix(){
	MatDuplicate(M, MAT_COPY_VALUES, &s1SysMatrix);
	MatAXPY(s1SysMatrix, -deltaT, C, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(s1SysMatrix, -deltaT*0.001, A, DIFFERENT_NONZERO_PATTERN);
	// std::cout << "M: " << std::endl;
	// MatView(M, PETSC_VIEWER_STDOUT_WORLD);
	// MatView(C, PETSC_VIEWER_STDOUT_WORLD);
	// MatView(A, PETSC_VIEWER_STDOUT_WORLD);	
	// MatView(s1SysMatrix, PETSC_VIEWER_STDOUT_WORLD);	
};

void stokesSolver::stage_two_matrix(){
	// Mat workspace;	
	// MatMatMult(A, D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &workspace);
	// MatTransposeMatMult(D, workspace, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &s2SysMatrixCond);
	// MatDestroy(&workspace);
	MatDuplicate(A, MAT_COPY_VALUES, &s2SysMatrix);
}


void stokesSolver::updateCmatrix(){
	MatDestroy(&C);
	C_matrix();
	MatDestroy(&s1SysMatrix);
	MatDuplicate(M, MAT_COPY_VALUES, &s1SysMatrix);
	MatAXPY(s1SysMatrix, -deltaT, C, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(s1SysMatrix, -deltaT*0.001, A, DIFFERENT_NONZERO_PATTERN);
	// std::cout << "M: " << std::endl;
	// MatView(M, PETSC_VIEWER_STDOUT_WORLD);
	// MatView(C, PETSC_VIEWER_STDOUT_WORLD);
	// MatView(A, PETSC_VIEWER_STDOUT_WORLD);	
	// MatView(s1SysMatrix, PETSC_VIEWER_STDOUT_WORLD);	
}

void stokesSolver::load_vector_element(double x1[], double x2[], double v[], int dir){
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
			gpPoints1[5*i+j] = fsource(x, y, dir);
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

void stokesSolver::load_vector(int dir){
	double v_element[4];
	double element_points[2][4];
	VecSet(V, 0.0);
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		load_vector_element(element_points[0], element_points[1], v_element, dir);
		for (int k = 0; k < 4; k++){
			VecSetValue(V, quadrilaterals[k][i], v_element[k], ADD_VALUES);
		}
	}
	VecAssemblyBegin(V);
	VecAssemblyEnd(V);
};

void stokesSolver::apply_boundary_condition(Vec RhsVector){
	Vec workspace;
	VecDuplicate(RhsVector, &workspace);
	VecZeroEntries(workspace);
	MatZeroRows(M, 4*division_Q1isoQ2, Bs.get(), 1.0, workspace, RhsVector);
	VecDestroy(&workspace);
	// MatZeroRowsColumns(finalMatrix, 2*4*division_Q1isoQ2, Bs.get(), 1.0, states, finalRhs);
};

void stokesSolver::linear_system_setup(){
	M_matrix();
	A_matrix();
	// D_matrix();
	B1_matrix();
	B2_matrix();
	C_matrix();
	stage_one_matrix();
	stage_two_matrix();
}

void stokesSolver::forwardEqn(){
	Vec rhs1, rhs2, rhs3;
	VecDuplicate(statesx, &rhs1);
	VecDuplicate(statesx, &rhs2);
	VecDuplicate(statesx, &rhs3);

	//x momentum
	load_vector(0);
	MatMult(s1SysMatrix, statesx, rhs1);
	VecDuplicate(V, &rhs2);
	VecCopy(V, rhs2);
	MatMult(B1, statesp, rhs3);
	VecAXPY(rhs1, deltaT, rhs2);
	VecAXPY(rhs1, deltaT, rhs3);
	apply_boundary_condition(rhs1);
	KSPSolve(kspm, rhs1, statesx);
	// VecView(statesx, PETSC_VIEWER_STDOUT_WORLD);

	//y momentum
	load_vector(1);
	MatMult(s1SysMatrix, statesy, rhs1);
	VecDuplicate(V, &rhs2);
	VecCopy(V, rhs2);
	MatMult(B2, statesp, rhs3);
	VecAXPBY(rhs1, deltaT, deltaT, rhs2);
	VecAXPY(rhs1, deltaT, rhs3);
	apply_boundary_condition(rhs1);
	KSPSolve(kspm, rhs1, statesy);
	VecDestroy(&rhs1);
	VecDestroy(&rhs2);
	VecDestroy(&rhs3);
	// VecView(statesy, PETSC_VIEWER_STDOUT_WORLD);

	//Poisson
	VecDuplicate(statesp, &rhs1);
	VecDuplicate(statesp, &rhs2);
	VecDuplicate(statesp, &rhs3);
	MatMult(B1, statesx, rhs1);
	MatMult(B2, statesy, rhs2);
	VecAXPBY(rhs1, -1.0/deltaT, -1.0/deltaT, rhs2);
	KSPSolve(kspa, rhs1, rhs3);
	VecAXPY(statesp, 1.0, rhs3);
	VecDestroy(&rhs1);
	VecDestroy(&rhs2);
	VecView(statesp, PETSC_VIEWER_STDOUT_WORLD);

	//velocity x update
	VecDuplicate(statesx, &rhs1);
	VecDuplicate(statesx, &rhs2);
	MatMult(M, statesx, rhs1);
	MatMult(B1, rhs3, rhs2);
	VecAXPY(rhs1, -deltaT, rhs2);
	apply_boundary_condition(rhs1);
	KSPSolve(kspm, rhs1, statesx);
	VecView(statesx, PETSC_VIEWER_STDOUT_WORLD);

	//velocity y update
	MatMult(M, statesy, rhs1);
	MatMult(B2, rhs3, rhs2);
	VecAXPY(rhs1, -deltaT, rhs2);
	apply_boundary_condition(rhs1);
	KSPSolve(kspm, rhs1, statesy);
	VecDestroy(&rhs1);
	VecDestroy(&rhs2);	
	VecDestroy(&rhs3);
	// VecView(statesy, PETSC_VIEWER_STDOUT_WORLD);

	VecGetArray(statesx, &states_array);
	std::copy(states_array, states_array+num_node_Q1isoQ2, u1.get());
	VecGetArray(statesy, &states_array);
	std::copy(states_array, states_array+num_node_Q1isoQ2, u2.get());
};

void stokesSolver::solve()
{
	for (int i = 0; i < timeSteps; ++i){
		forwardEqn();
		updateCmatrix();
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


void writeU(double u1[], double u2[], int arraySize, std::string pathName){
	std::ofstream myfile;
	myfile.open(pathName);
	for (int i = 0; i < arraySize; ++i){
		myfile << u1[i] << " " << u2[i] << std::endl;
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
	double pCNstep = std::stod(value[num_term+3]);
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
		std::cout << "levels: "           << levels       <<  std::endl;
		std::cout << "a: "                << a            <<  std::endl;
		std::cout << "pCNstep: "          << pCNstep          <<  std::endl;
		std::cout << "task: "             << task         <<  std::endl;
		std::cout << "plainMCMC samples:" << plainMCMC_sample_number << std::endl;
		std::cout << "obsNumPerRow: "     << obsNumPerRow <<  std::endl;
		std::cout << "noiseVariance: "    << noiseVariance << std::endl; 
	}

	if (task == 8){
		stokesSolver testSolver(levels, 1, 1);
		testSolver.samples[0] = 1.0;
		testSolver.solve();
	}

	PetscFinalize();
	return 0;
}
