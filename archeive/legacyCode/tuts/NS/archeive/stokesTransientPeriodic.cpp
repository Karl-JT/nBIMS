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
#include <MLMCMC_Bi_Uniform.h>
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
	std::unique_ptr<int[]> PBC_Q1isoQ2_idx;
	std::unique_ptr<int[]> PBC_Q1_idx;
	std::unique_ptr<int[]> bs;

	Mat M1;
	Mat M2;
	Mat A1;
	Mat A2;
	Mat B1;
	Mat B1T;
	Mat B2;
	Mat B2T;
	Mat D;
	Mat sysMatrix;
	Mat massMatrix;
	Vec V;
	Vec finalRhs;

	KSP ksp;
	PC  pc;

public:
	Vec states;
	Vec intVecObs;
	Vec intVecQoi;
	IS isRhsVector;
	IS isrow;
	IS iscol;

	Mat finalMatrix;
	Mat timeMatrix;

	int division0 = 2;
	int division_Q1isoQ2;
	int division_Q1;
	int num_Q1isoQ2_element;
	int num_Q1_element;
	int num_node_Q1isoQ2;
	int num_node_Q1;

	std::unique_ptr<std::unique_ptr<double[]>[]> points;

	int level;
	int num_term;
	double obs=0.184531597719236;
	double currentObs;
	double qoi;
	double noiseVariance;
	double beta = 1.0;
	double *states_array;
	double *adjoint_array;
	double delta_t;
	std::unique_ptr<double[]> samples;
	std::unique_ptr<double[]> u1;
	std::unique_ptr<double[]> u2;

	std::default_random_engine generator;
	// std::normal_distribution<double> normalDistribution{0.0, 1.0};
    std::uniform_real_distribution<double> uniformDistribution{-1.0, 1.0};

	stokesSolver(int level_, int num_term_, double noiseVariance_);
	~stokesSolver(){
		ISDestroy(&isRhsVector);
		ISDestroy(&isrow);
		ISDestroy(&iscol);
		MatDestroy(&M1);
		MatDestroy(&M2);
		MatDestroy(&A1);
		MatDestroy(&A2);
		MatDestroy(&B1);
		MatDestroy(&B1T);
		MatDestroy(&B2);
		MatDestroy(&B2T);
		MatDestroy(&D);
		MatDestroy(&sysMatrix);
		MatDestroy(&massMatrix);
		MatDestroy(&finalMatrix);
		MatDestroy(&timeMatrix);
		VecDestroy(&finalRhs);		
		VecDestroy(&states);		
		VecDestroy(&intVecObs);		
		VecDestroy(&intVecQoi);		
		VecDestroy(&V);		
		KSPDestroy(&ksp);
	};

	void createAllocations();
	double fsource(double x, double y);
	void shape_function(double epsilon, double eta, double N[]);
	double shape_interpolation(double epsilon, double eta, double x[4]);
	void jacobian_matrix(double x1[4], double x2[4], double epsilon, double eta, double J[2][2]);
	double jacobian_det(double J[2][2]);
	void jacobian_inv(double x1[4], double x2[4], double epsilon, double eta, double J[2][2]);
	void basis_function(double epsilon, double eta, double N[]);
	void basis_function_derivative(double dPhi[4][2], double epsilon, double eta);
	void hat_function_derivative(double dPhi[4][2], double epsilon, double eta, double x1[4], double x2[4]);
	void mass_matrix_element(double P[][4], double A[][4]);
	void mass_matrix();
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
	void int_vector_element(double x1[], double x2[], double vx[], double vy[], double expCoef);
	void int_vector_obs(double expCoef);
	void int_vector_qoi(double expCoef);
	double qoiOutput();
	double obsOutput();
	void system_rhs();
	// void apply_boundary_condition();
	// void apply_homo_boundary_condition();
	void linear_system_setup();
	void pointObsMatrix(double obs[][4], int size);
	void forwardEqn();
	void updateGeneratorSeed(double seed_);

	void solve();
	double lnLikelihood();
	double solve4Obs();
	double solve4QoI();
	double getAlpha(double lnLikelihoodt0, double lnLikelihoodt1);
	void priorSample(double initialSamples[]);

};

stokesSolver::stokesSolver(int level_, int num_term_, double noiseVariance_) : level(level_), num_term(num_term_), noiseVariance(noiseVariance_) {
	samples = std::make_unique<double[]>(num_term_);
	timeSteps = std::pow(2, level_+1);
	delta_t = 1.0/(2.0*(std::pow(2.0, level_+1.0)));

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

	//Q1isoQ2 mapping
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

	//Periodic Boundary Condition Mapping
	std::vector<int> PBC_Q1isoQ2(num_node_Q1isoQ2);
	std::vector<int> PBC_Q1(num_node_Q1isoQ2);

	std::iota(PBC_Q1isoQ2.begin(), PBC_Q1isoQ2.end(), 0);
	for (int i = 0; i < num_node_Q1isoQ2; ++i){
		PBC_Q1[i] = mesh_idx[i];
	}

	bs = std::make_unique<int[]>(division_Q1isoQ2*2+1);
	int b_pos[2] = {division_Q1isoQ2, num_node_Q1isoQ2-1};
	bs[0] = b_pos[0];
	bs[1] = b_pos[1];
	bs[division_Q1isoQ2*2] = num_node_Q1isoQ2-1-division_Q1isoQ2;	 
	PBC_Q1isoQ2[b_pos[0]] = 0;
	PBC_Q1isoQ2[b_pos[1]] = 0;
	PBC_Q1isoQ2[num_node_Q1isoQ2-1-division_Q1isoQ2] = 0;
	PBC_Q1[b_pos[0]] = mesh_idx[0];
	PBC_Q1[b_pos[1]] = mesh_idx[0];
	PBC_Q1[num_node_Q1isoQ2-1-division_Q1isoQ2] = mesh_idx[0];
	for (int i = 1; i < division_Q1isoQ2;++i){
		PBC_Q1isoQ2[b_pos[0]+(division_Q1isoQ2+1)*i] = b_pos[0]+(division_Q1isoQ2+1)*i - division_Q1isoQ2;
		PBC_Q1isoQ2[b_pos[1]-i] = division_Q1isoQ2 - i;
		PBC_Q1[b_pos[0]+(division_Q1isoQ2+1)*i] = mesh_idx[b_pos[0]+(division_Q1isoQ2+1)*i - division_Q1isoQ2];
		PBC_Q1[b_pos[1]-i] = mesh_idx[division_Q1isoQ2 - i];
		bs[2*i] = b_pos[0]+(division_Q1isoQ2+1)*i;
		bs[2*i+1] = b_pos[1]-i;
	}

	PBC_Q1isoQ2_idx = std::make_unique<int[]>(num_node_Q1isoQ2);
	PBC_Q1_idx = std::make_unique<int[]>(num_node_Q1isoQ2);
	for (int i = 0; i < num_node_Q1isoQ2; i++){
		PBC_Q1isoQ2_idx[i] = PBC_Q1isoQ2[i];
		PBC_Q1_idx[i] = PBC_Q1[i];
	}	

	createAllocations();

	linear_system_setup();
	system_rhs();

	KSPCreate(PETSC_COMM_SELF, &ksp);
	KSPSetType(ksp, KSPGMRES);
	KSPSetOperators(ksp, finalMatrix, finalMatrix);
	KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCFIELDSPLIT);
	PCFieldSplitSetDetectSaddlePoint(pc, PETSC_TRUE);
	PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELF, NULL);
	PCSetUp(pc);

	////////////////Not Nested/////////////////////////////
	KSP      *sub_ksp;
	PC       sub_pc_0;
	PetscInt nsplits;
	PCFieldSplitSchurGetSubKSP(pc, &nsplits, &sub_ksp);
	KSPGetPC(sub_ksp[0], &sub_pc_0);
	PCSetType(sub_pc_0, PCILU);
	PCSetUp(sub_pc_0);
	KSPSetType(sub_ksp[0], KSPPREONLY);
	KSPSetUp(sub_ksp[0]);


	KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	// KSPMonitorSet(ksp, MyKSPMonitor, NULL, 0);
	KSPSetUp(ksp);
	KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
	PetscFree(sub_ksp);

	u1 = std::make_unique<double[]>(num_node_Q1isoQ2);
	u2 = std::make_unique<double[]>(num_node_Q1isoQ2);

	int_vector_obs(0.5);
	int_vector_qoi(0.5);
};

void stokesSolver::createAllocations(){
 	VecCreate(MPI_COMM_SELF, &finalRhs);
	VecSetSizes(finalRhs, PETSC_DECIDE, 2*num_node_Q1isoQ2+num_node_Q1);
	if (level > 3){
		VecSetType(finalRhs, VECSEQCUDA);
	} else {
		VecSetType(finalRhs, VECSEQ);
	}

 	VecCreate(MPI_COMM_SELF, &states);
	VecSetSizes(states, PETSC_DECIDE, 2*num_node_Q1isoQ2+num_node_Q1);
	if (level > 3){
		VecSetType(states, VECSEQCUDA);
	} else {
		VecSetType(states, VECSEQ);
	}

 	VecCreate(MPI_COMM_SELF, &V);
	VecSetSizes(V, PETSC_DECIDE, 3*num_node_Q1isoQ2);
	if (level > 3){
		VecSetType(V, VECSEQCUDA);
	} else {
		VecSetType(V, VECSEQ);
	}
	// VecSetFromOptions(V);

 	VecCreate(MPI_COMM_SELF, &intVecQoi);
	VecSetSizes(intVecQoi, PETSC_DECIDE, 2*num_node_Q1isoQ2+num_node_Q1);
	if (level > 3){
		VecSetType(intVecQoi, VECSEQCUDA);
	} else {
		VecSetType(intVecQoi, VECSEQ);
	}
	// VecSetFromOptions(intVecQoi);

 	VecCreate(MPI_COMM_SELF, &intVecObs);
	VecSetSizes(intVecObs, PETSC_DECIDE, 2*num_node_Q1isoQ2+num_node_Q1);
	if (level > 3){
		VecSetType(intVecObs, VECSEQCUDA);
	} else {
		VecSetType(intVecObs, VECSEQ);
	}
	// VecSetFromOptions(intVecObs);

	MatCreate(PETSC_COMM_SELF, &M1);
	MatSetSizes(M1, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(M1, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(M1, MATSEQAIJ);
	}
	// MatSetFromOptions(M1);
	MatSeqAIJSetPreallocation(M1, 12, NULL);
	MatMPIAIJSetPreallocation(M1, 12, NULL, 12, NULL);

	MatCreate(PETSC_COMM_SELF, &M2);
	MatSetSizes(M2, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(M2, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(M2, MATSEQAIJ);
	}
	// MatSetFromOptions(A1);
	MatSeqAIJSetPreallocation(M2, 12, NULL);
	MatMPIAIJSetPreallocation(M2, 12, NULL, 12, NULL);

	MatCreate(PETSC_COMM_SELF, &A1);
	MatSetSizes(A1, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(A1, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(A1, MATSEQAIJ);
	}
	// MatSetFromOptions(A1);
	MatSeqAIJSetPreallocation(A1, 12, NULL);
	MatMPIAIJSetPreallocation(A1, 12, NULL, 12, NULL);

	MatCreate(PETSC_COMM_SELF, &A2);
	MatSetSizes(A2, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(A2, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(A2, MATSEQAIJ);
	}
	// MatSetFromOptions(A2);
	MatSeqAIJSetPreallocation(A2, 12, NULL);
	MatMPIAIJSetPreallocation(A2, 12, NULL, 12, NULL);

	MatCreate(PETSC_COMM_SELF, &B1);
	MatSetSizes(B1, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(B1, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(B1, MATSEQAIJ);
	}
	// MatSetFromOptions(B1);
	MatSeqAIJSetPreallocation(B1, 24, NULL);
	MatMPIAIJSetPreallocation(B1, 24, NULL, 24, NULL);

	MatCreate(PETSC_COMM_SELF, &B1T);
	MatSetSizes(B1T, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(B1T, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(B1T, MATSEQAIJ);
	}
	// MatSetFromOptions(B1T);
	MatSeqAIJSetPreallocation(B1T, 24, NULL);
	MatMPIAIJSetPreallocation(B1T, 24, NULL, 24, NULL);


	MatCreate(MPI_COMM_SELF, &B2);
	MatSetSizes(B2, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(B2, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(B2, MATSEQAIJ);
	}
	// MatSetFromOptions(B2);
	MatSeqAIJSetPreallocation(B2, 24, NULL);
	MatMPIAIJSetPreallocation(B2, 24, NULL, 24, NULL);

	MatCreate(MPI_COMM_SELF, &B2T);
	MatSetSizes(B2T, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(B2T, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(B2T, MATSEQAIJ);
	}
	// MatSetFromOptions(B2T);
	MatSeqAIJSetPreallocation(B2T, 24, NULL);
	MatMPIAIJSetPreallocation(B2T, 24, NULL, 24, NULL);

	int final_size = 2*num_node_Q1isoQ2+num_node_Q1;
	PetscInt *indices;
	PetscMalloc1(final_size, &indices);
	for (int i = 0; i < final_size; i++){
		indices[i] = i;
	}
	ISCreateGeneral(PETSC_COMM_SELF, final_size, indices, PETSC_COPY_VALUES, &isRhsVector);
	ISCreateGeneral(PETSC_COMM_SELF, final_size, indices, PETSC_COPY_VALUES, &isrow);
	ISCreateGeneral(PETSC_COMM_SELF, final_size, indices, PETSC_COPY_VALUES, &iscol);
	PetscFree(indices);
}

void stokesSolver::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};

double stokesSolver::fsource(double x, double y){
    double output;
    output = samples[0]*(std::sin(2.*M_PI*x)+std::sin(2.*M_PI*y)); 
    /*        + u[1]*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y) \
            + u[2]*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y) \
            + u[3]*std::cos(2.*M_PI*x)*std::cos(2.*M_PI*y) ;
    */
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
	double Jdet[25];

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

	double gpPoints[25][4];
	double gpWeights[25];

	for (int i = 0; i < 5; i++){
		for (int j = 0; j < 5; j++){
			basis_function(refPoints[i], refPoints[j], N);
			jacobian_matrix(P[0], P[1], refPoints[i], refPoints[j], J);
			gpWeights[5*i+j] = refWeights[i]*refWeights[j];
			Jdet[5*i+j] = jacobian_det(J);
			for (int k = 0; k < 4; ++k){
				gpPoints[5*i+j][k] = N[k];
			}
		}
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			A[i][j] = 0;
			for (int k = 0; k < 25; k++){
				A[i][j] += gpWeights[k]*gpPoints[k][i]*gpPoints[k][j]*Jdet[k];
			}
		}
	}
}

void stokesSolver::mass_matrix(){
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
 				MatSetValue(M1, PBC_Q1isoQ2_idx[quadrilaterals[m][i]], PBC_Q1isoQ2_idx[quadrilaterals[n][i]], mass_element[m][n], ADD_VALUES);
			}
		}			
	}
	MatAssemblyBegin(M1, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(M1, MAT_FINAL_ASSEMBLY);


	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		mass_matrix_element(element_points, mass_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
 				MatSetValue(M2, num_node_Q1isoQ2+PBC_Q1isoQ2_idx[quadrilaterals[m][i]], num_node_Q1isoQ2+PBC_Q1isoQ2_idx[quadrilaterals[n][i]], mass_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(M2, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(M2, MAT_FINAL_ASSEMBLY);
}

void stokesSolver::stiffness_matrix_element(double P[][4], double A[][4]){
	double dPhi[4][2];
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
				A[i][j] += 0.0001*gpWeights[k]*(gpPoints[k][i]*gpPoints[k][j]+gpPoints[k][4+i]*gpPoints[k][4+j])*Jdet[k];
			}
		}
	}
};

void stokesSolver::stiffness_matrix(){
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
 				MatSetValue(A1, PBC_Q1isoQ2_idx[quadrilaterals[m][i]], PBC_Q1isoQ2_idx[quadrilaterals[n][i]], stiffness_element[m][n], ADD_VALUES);
			}
		}			
	}
	for (int i = 0; i < 3*num_node_Q1isoQ2; ++i){
		MatSetValue(A1, i, i, 0, ADD_VALUES);
	}
	MatAssemblyBegin(A1, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A1, MAT_FINAL_ASSEMBLY);

	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		stiffness_matrix_element(element_points, stiffness_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
 				MatSetValue(A2, num_node_Q1isoQ2+PBC_Q1isoQ2_idx[quadrilaterals[m][i]], num_node_Q1isoQ2+PBC_Q1isoQ2_idx[quadrilaterals[n][i]], stiffness_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(A2, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A2, MAT_FINAL_ASSEMBLY);
};

void stokesSolver::b1_matrix_element(double x1[4], double x2[4], double b1[4][4]){
	double dPhi[4][2];
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
				MatSetValue(B1, 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[m][i]], PBC_Q1isoQ2_idx[quadrilaterals[n][i]], b1_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(B1, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B1, MAT_FINAL_ASSEMBLY);	

	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		b1_matrix_element(element_points[0], element_points[1], b1_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
				MatSetValue(B1T, PBC_Q1isoQ2_idx[quadrilaterals[n][i]], 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[m][i]], b1_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(B1T, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B1T, MAT_FINAL_ASSEMBLY);	
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
				MatSetValue(B2, 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[m][i]], num_node_Q1isoQ2+PBC_Q1isoQ2_idx[quadrilaterals[n][i]], b2_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(B2, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B2, MAT_FINAL_ASSEMBLY);	

	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}

		b2_matrix_element(element_points[0], element_points[1], b2_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
				MatSetValue(B2T, num_node_Q1isoQ2+PBC_Q1isoQ2_idx[quadrilaterals[n][i]], 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[m][i]], b2_element[m][n], ADD_VALUES);
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

	MatAXPY(M1, 1, M2, DIFFERENT_NONZERO_PATTERN);
};

void stokesSolver::Q1isoQ2Cond(){
	MatCreate(MPI_COMM_SELF, &D);
	MatSetSizes(D, PETSC_DECIDE, PETSC_DECIDE, 3*num_node_Q1isoQ2, 3*num_node_Q1isoQ2);
	if (level > 3){
		MatSetType(D, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(D, MATSEQAIJ);
	}
	// MatSetFromOptions(D);
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
			MatSetValue(D, 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[1][i]], 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[0][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[3][i]], 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[0][i]], 0.5, INSERT_VALUES);
		} else if (Q1_idx[quadrilaterals[1][i]] < num_node_Q1isoQ2){
			MatSetValue(D, 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[2][i]], 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[1][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[0][i]], 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[1][i]], 0.5, INSERT_VALUES);			
		} else if (Q1_idx[quadrilaterals[2][i]] < num_node_Q1isoQ2){
			MatSetValue(D, 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[3][i]], 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[2][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[1][i]], 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[2][i]], 0.5, INSERT_VALUES);			
		} else if (Q1_idx[quadrilaterals[3][i]] < num_node_Q1isoQ2){
			MatSetValue(D, 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[0][i]], 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[3][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[2][i]], 2*num_node_Q1isoQ2+PBC_Q1_idx[quadrilaterals[3][i]], 0.5, INSERT_VALUES);			
		}
	}
	MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);

	Mat workspace;	
	MatMatMult(A1, D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &workspace);
	MatTransposeMatMult(D, workspace, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &sysMatrix);
	
	MatMatMult(M1, D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &workspace);
	MatTransposeMatMult(D, workspace, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &massMatrix);
	MatScale(massMatrix, 1.0/delta_t);

	MatDestroy(&workspace);

	MatCreateSubMatrix(sysMatrix, isrow, iscol, MAT_INITIAL_MATRIX, &finalMatrix);
	MatCreateSubMatrix(massMatrix, isrow, iscol, MAT_INITIAL_MATRIX, &timeMatrix);
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
	double v_element[4];
	double element_points[2][4];
	VecSet(V, 0.0);
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		load_vector_element(element_points[0], element_points[1], v_element);
		for (int k = 0; k < 4; k++){
			VecSetValue(V, PBC_Q1isoQ2_idx[quadrilaterals[k][i]], v_element[k], ADD_VALUES);
			VecSetValue(V, num_node_Q1isoQ2+PBC_Q1isoQ2_idx[quadrilaterals[k][i]], v_element[k], ADD_VALUES);
		}
	}
	VecAssemblyBegin(V);
	VecAssemblyEnd(V);
};


void stokesSolver::int_vector_element(double x1[], double x2[], double vx[], double vy[], double expCoef){
	double dPhi[4][2];
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

	double gpPoints1x[25];
	double gpPoints1y[25];
	double gpPoints2x[25][4];
	double gpPoints2y[25][4];
	double gpWeights[25];
	double Jdet[25];

	double x;
	double y;

	for (int i = 0; i < 5; i++){
		for (int j = 0; j < 5; j++){
			hat_function_derivative(dPhi, refPoints[i], refPoints[j], x1, x2);
			jacobian_matrix(x1, x2, refPoints[i], refPoints[j], J);
			x = shape_interpolation(refPoints[i], refPoints[j], x1);
			y = shape_interpolation(refPoints[i], refPoints[j], x2);
			gpWeights[5*i+j] = refWeights[i]*refWeights[j];
			Jdet[5*i+j] = jacobian_det(J);
			gpPoints1x[5*i+j] = pow(x, expCoef);
			gpPoints1y[5*i+j] = pow(y, expCoef);
			for (int k = 0; k < 4; ++k){
				gpPoints2x[5*i+j][k] = dPhi[k][0];
				gpPoints2y[5*i+j][k] = dPhi[k][1];
			}
		}
	}
	for (int i = 0; i < 4; i++){
		vx[i] = 0;
		vy[i] = 0;
		for (int k = 0; k < 25; k++){
			vx[i] += gpWeights[k]*gpPoints1y[k]*gpPoints2y[k][i]*Jdet[k];
			vy[i] += gpWeights[k]*gpPoints1x[k]*gpPoints2x[k][i]*Jdet[k];
		}
	}
};

void stokesSolver::int_vector_obs(double expCoef){
	double v_element_x[4];
	double v_element_y[4];
	double element_points[2][4];
	VecSet(intVecObs, 0.0);
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		int_vector_element(element_points[0], element_points[1], v_element_x, v_element_y, expCoef);
		for (int k = 0; k < 4; k++){
			VecSetValue(intVecObs, PBC_Q1isoQ2_idx[quadrilaterals[k][i]], v_element_x[k], ADD_VALUES);
			VecSetValue(intVecObs, num_node_Q1isoQ2+PBC_Q1isoQ2_idx[quadrilaterals[k][i]], v_element_y[k], ADD_VALUES);
		}
	}

	VecAssemblyBegin(intVecObs);
	VecAssemblyEnd(intVecObs);
};

void stokesSolver::int_vector_qoi(double expCoef){
	double v_element_x[4];
	double v_element_y[4];
	double element_points[2][4];
	VecSet(intVecQoi, 0.0);
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		int_vector_element(element_points[0], element_points[1], v_element_x, v_element_y, expCoef);
		for (int k = 0; k < 4; k++){
			VecSetValue(intVecQoi, PBC_Q1isoQ2_idx[quadrilaterals[k][i]], v_element_x[k], ADD_VALUES);
			VecSetValue(intVecQoi, num_node_Q1isoQ2+PBC_Q1isoQ2_idx[quadrilaterals[k][i]], v_element_y[k], ADD_VALUES);
		}
	}
	VecAssemblyBegin(intVecQoi);
	VecAssemblyEnd(intVecQoi);
};

void stokesSolver::system_rhs(){
	load_vector();
	Vec rhs;
	VecGetSubVector(V, isRhsVector, &rhs);
	VecCopy(rhs, finalRhs);
	VecDestroy(&rhs);

	MatMultAdd(timeMatrix, states, finalRhs, finalRhs);
};

void stokesSolver::linear_system_setup(){
	stiffness_matrix();
	mass_matrix();
	B1_matrix();
	B2_matrix();
	system_matrix();
	Q1isoQ2Cond();	
	MatAXPY(finalMatrix, 1.0, timeMatrix, DIFFERENT_NONZERO_PATTERN);
}

void stokesSolver::forwardEqn(){
	system_rhs();

	KSPSolve(ksp, finalRhs, states);
	// VecView(states, PETSC_VIEWER_STDOUT_WORLD);
	VecGetArray(states, &states_array);
	std::copy(states_array, states_array+num_node_Q1isoQ2, u1.get());
	std::copy(states_array+num_node_Q1isoQ2, states_array+2*num_node_Q1isoQ2, u2.get());
};

// void stokesSolver::priorSample(double initialSamples[]){
// 	initialSamples[0] = normalDistribution(generator);
// };

void stokesSolver::priorSample(double initialSamples[]){
	initialSamples[0] = uniformDistribution(generator);
}

double stokesSolver::obsOutput(){
	return currentObs;
}

double stokesSolver::qoiOutput(){
	return qoi;
}

void stokesSolver::solve()
{
	VecZeroEntries(states);
	for (int  i = 0; i < 1/delta_t; ++i){
 		forwardEqn();
		if (i == 0.5/delta_t-1){
			int_vector_qoi(0.5);
			VecDot(intVecQoi, states, &qoi);		
			std::cout << "qoi: " << qoi << std::endl;	
		}
	int_vector_obs(0.5);
	VecDot(intVecObs, states, &currentObs);
	std::cout << "obs: " << currentObs << std::endl;	
	}
	// VecView(states, PETSC_VIEWER_STDOUT_WORLD);
}

double stokesSolver::lnLikelihood(){
	double obsResult = obsOutput();
	double lnlikelihood = -0.5/noiseVariance*pow(obsResult-obs, 2);
	return lnlikelihood;
}

double stokesSolver::solve4Obs(){
	return obsOutput();
}

double stokesSolver::solve4QoI(){
	return qoiOutput();
}

double stokesSolver::getAlpha(double lnLikelihoodt0, double lnLikelihoodt1){
	double a = std::min(0.0, lnLikelihoodt1-lnLikelihoodt0);
	return a;
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


	if (task == 2){
		double output;
		MLMCMC_Bi_Uniform<pCN<stokesSolver>, stokesSolver> MLMCMCSolver(levels, 1, rank, a, noiseVariance, 1.0);
		output = MLMCMCSolver.mlmcmcRun();

		// MLMCMC_Bi<pCN<stokesSolver>, stokesSolver> MLMCMCSolver(levels, 1, rank, a, noiseVariance, 1.0);
		// output = MLMCMCSolver.mlmcmcRun();
		std::cout << output << std::endl;

		std::string outputfile = "output_";
		outputfile.append(std::to_string(rank));

		std::ofstream myfile;
		myfile.open(outputfile);
		for (int i = 0; i<num_term; ++i){
			myfile << output << " ";
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


	if (task == 6){ //Generate Reference
		double quadx[128]; //= {-0.9894, -0.9446, -0.8656, -0.7554, -0.6179, -0.4580, -0.2816, -0.0950,   0.0950,    0.2816,    0.4580,    0.6179,   0.7554,  0.8656,    0.9446,    0.9894};
		double quadw[128]; //= {0.0272, 0.0623, 0.0952, 0.1246, 0.1496, 0.1692, 0.1826, 0.1895, 0.1895,    0.1826,    0.1692,    0.1496,    0.1246, 0.0952,    0.0623,    0.0272};
		
		double obs;
		double qoi;

		double localNum=0;
		double localDen=0;
 
		// gauleg(64, quadx, quadw);  //gaussian legendre
		gauher2(128, quadx, quadw);     //gaussian hermite

		stokesSolver refSolver(levels, 1, 1);

		for (int i = 0; i < 4; ++i){
			refSolver.samples[0] = quadx[rank+16*i];
			std::cout << "rank: " << rank << " coef: " << refSolver.samples[0] << std::endl;
			refSolver.forwardEqn();

			refSolver.int_vector_obs(1.5);
			obs = refSolver.obsOutput();

			refSolver.int_vector_qoi(0.5);
			qoi = refSolver.qoiOutput();

			localNum += qoi*quadw[rank+i*16]*exp(-10*pow(obs+0.994214, 2)/2)+(-qoi)*quadw[rank+i*16]*exp(-10*pow(-obs+0.994214, 2)/2);
			localDen += quadw[rank+i*16]*exp(-10*pow(obs+0.994214, 2)/2)+quadw[rank+i*16]*exp(-10*pow(-obs+0.994214, 2)/2);

			std::cout << "rank: " << rank << " obs: " << obs << " qoi: " << qoi << std::endl;
		}

		double globalNum, globalDen;
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&localNum, &globalNum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&localDen, &globalDen, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (rank == 0){
			std::cout.precision(10);
			std::cout << globalNum << " " << globalDen << " " << globalNum/globalDen << std::endl;
		}
	}

	if (task == 8){
		double obs1;
		double qoi1;
		stokesSolver testSolver(levels, 1, 1);		
		testSolver.samples[0] = 0.5152;
		testSolver.solve();
		obs1 = testSolver.obsOutput();
		qoi1 = testSolver.qoiOutput();
		std::cout << obs1 << " " << qoi1 << std::endl;
	}

	PetscFinalize();
	return 0;
}
