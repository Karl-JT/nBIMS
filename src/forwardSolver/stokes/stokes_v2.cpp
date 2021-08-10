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

#include "FEModule.cpp"

PetscErrorCode MyKSPMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *dummy)
{
	PetscPrintf(PETSC_COMM_SELF, "iteration %D KSP Residual norm %14.12e \n", n, rnorm);
	return 0;
}

int main(int argc, char* argv[]){
	PetscInitialize(&argc, &argv, NULL, NULL);

	int rank, size;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	Mesh L2;
	PetscNew(&L2);
	L2->comm = PETSC_COMM_WORLD;
	assembleSystem(L2, 5);

	applyBoundary(L2);

	// MatView(L2->system, PETSC_VIEWER_STDOUT_WORLD);
	// VecView(L2->rhs, PETSC_VIEWER_STDOUT_WORLD);


	KSP ksp;
	PC  pc;
	KSPCreate(L2->comm, &ksp);
	KSPSetType(ksp, KSPFGMRES);
	KSPSetOperators(ksp, L2->system, L2->system);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCFIELDSPLIT);
	PCFieldSplitSetDetectSaddlePoint(pc, PETSC_TRUE);
	PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, NULL);
	// PCFieldSplitSetBlockSize(pc, 3);
	// PCFieldSplitSetFields(pc, "0", 2, ufields, ufields);
	// PCFieldSplitSetFields(pc, "1", 1, pfields, pfields);
	// PCSetUp(pc);
	// PCView(pc, PETSC_VIEWER_STDOUT_WORLD);
	// PCFieldSplitGetIS(pc, "0", &is[0]);
	// PCFieldSplitGetIS(pc, "1", &is[1]);
	// MatCreateSubMatrix(L2->system, is[0], is[0], MAT_INITIAL_MATRIX, &A00);
	// MatCreateSubMatrix(L2->system, is[0], is[1], MAT_INITIAL_MATRIX, &A01);
	// MatCreateSubMatrix(L2->system, is[1], is[0], MAT_INITIAL_MATRIX, &A10);
	// MatCreateSubMatrix(L2->system, is[1], is[1], MAT_INITIAL_MATRIX, &A11);
	// MatConvert(A11, MATMPIAIJCUSPARSE, MAT_INITIAL_MATRIX, &A11cuda);
	// MatView(A11cuda, PETSC_VIEWER_STDOUT_WORLD);
	// MatView(L2->system, PETSC_VIEWER_STDOUT_WORLD);
	// VecView(L2->rhs, PETSC_VIEWER_STDOUT_WORLD);
	// PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, A11);
	PCSetUp(pc);

	
	KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPMonitorSet(ksp, MyKSPMonitor, NULL, 0);
	KSPSetUp(ksp);
	KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
	
	KSPSolve(ksp, L2->rhs, L2->solution);
	// VecGetArray(states, &states_array);
	// std::copy(states_array, states_array+num_node_Q1isoQ2, u1.get());
	// std::copy(states_array+num_node_Q1isoQ2, states_array+2*num_node_Q1isoQ2, u2.get());
	// VecView(L2->solution, PETSC_VIEWER_STDOUT_WORLD);

	PetscFinalize();
	return 0;
}



// class stokesSolver {
// private:
// 	std::unique_ptr<std::unique_ptr<int[]>[]> quadrilaterals;
// 	std::unique_ptr<int[]> Q1_idx;
// 	std::unique_ptr<int[]> mesh_idx;
// 	std::unique_ptr<int[]> Bs;

// 	Mat A1;
// 	Mat A2;
// 	Mat B1;
// 	Mat B1T;
// 	Mat B2;
// 	Mat B2T;
// 	Mat D;
// 	Mat I;
// 	Mat sysMatrix;
// 	Mat state2obsMatrix;
// 	Vec V;
// 	Vec rhs;
// 	Vec adjointRhs;
// 	Vec finalRhs;

// 	KSP ksp;
// 	PC  pc;

// public:
// 	Vec states;
// 	Vec adjoints;
// 	Vec intVec;

// 	Mat finalMatrix;

// 	int division0 = 2;
// 	int division_Q1isoQ2;
// 	int division_Q1;
// 	int num_Q1isoQ2_element;
// 	int num_Q1_element;
// 	int num_node_Q1isoQ2;
// 	int num_node_Q1;

// 	std::unique_ptr<std::unique_ptr<double[]>[]> points;

// 	int num_term;
// 	int level;
// 	double noiseVariance;
// 	double beta = 1;
// 	double *states_array;
// 	double *adjoint_array;
// 	std::unique_ptr<double[]> u;
// 	std::unique_ptr<double[]> u1;
// 	std::unique_ptr<double[]> u2;

// 	std::default_random_engine generator;
// 	std::normal_distribution<double> distribution{0.0, 1.0};

// 	stokesSolver(int level_, int num_term_, double noiseVariance_);
// 	~stokesSolver(){};

// 	double fsource(double x, double y);
// 	double fsource_i(double x, double y, int i);
// 	void shape_function(double epsilon, double eta, double N[]);
// 	double shape_interpolation(double epsilon, double eta, double x[4]);
// 	void jacobian_matrix(double x1[4], double x2[4], double epsilon, double eta, double J[2][2]);
// 	double jacobian_det(double J[2][2]);
// 	void jacobian_inv(double x1[4], double x2[4], double epsilon, double eta, double J[2][2]);
// 	void basis_function(double epsilon, double eta, double N[]);
// 	void basis_function_derivative(double dPhi[4][2], double epsilon, double eta);
// 	void hat_function_derivative(double dPhi[4][2], double epsilon, double eta, double x1[4], double x2[4]);
// 	void stiffness_matrix_element(double P[][4], double A[][4]);
// 	void stiffness_matrix();
// 	void b1_matrix_element(double x1[4], double x2[4], double b1[4][4]);
// 	void b2_matrix_element(double x1[4], double x2[4], double b1[4][4]);
// 	void B1_matrix();
// 	void B2_matrix();
// 	void system_matrix();
// 	void Q1isoQ2Cond();
// 	void load_vector_element(double x1[], double x2[], double v[]);
// 	void area_vector_element();
// 	void load_vector();
// 	void load_vector_element_i(double x1[], double x2[], double v[], int i);
// 	void load_vector_i(int i, double load[]);
// 	void int_vector_element(double x1[], double x2[], double vx[], double vy[]);
// 	void int_vector();
// 	void system_rhs();
// 	void apply_boundary_condition();
// 	void apply_homo_boundary_condition();
// 	void linear_system_setup();
// 	void pointObsMatrix(double obs[][4], int size);
// 	void forwardEqn();
// 	void adjointEqn(int size, double obs[][4]);
// 	void controlEqn(double grad[]);	
// 	void WuuApply(Vec direction, int size);
// 	Vec CApply(Vec direction, int size, bool T);
// 	Vec ASolve(Vec direction, int size, bool T);
// 	Vec RApply(Vec direction, bool T);
// 	void HessianAction(double grad[]);
// 	void fullHessian();
// 	void updateGeneratorSeed(double seed_);
// 	void pCNProposalGenerator(double proposal[]);
// 	void SNProposalGenerator(double proposal[]);
// 	double lnlikelihood(double obs[][4], int size);
// 	void getValues(double points[][2], double u[][2], int size);
// 	double solve(double obs[][4], int size);
// };

// stokesSolver::stokesSolver(int level_, int num_term_, double noiseVariance_) : level(level_), num_term(num_term_), noiseVariance(noiseVariance_) {
// 	u = std::make_unique<double[]>(num_term_);

// 	division_Q1isoQ2 = division0*(std::pow(2, level_+1));
// 	division_Q1 = division_Q1isoQ2/2;
// 	num_Q1isoQ2_element = division_Q1isoQ2*division_Q1isoQ2;
// 	num_Q1_element = 0.25*num_Q1isoQ2_element;
// 	num_node_Q1isoQ2 = std::pow(division_Q1isoQ2+1, 2);
// 	num_node_Q1 = std::pow(division_Q1+1, 2);

// 	points    = std::make_unique<std::unique_ptr<double[]>[]>(2);
// 	points[0] = std::make_unique<double[]>(num_node_Q1isoQ2);
// 	points[1] = std::make_unique<double[]>(num_node_Q1isoQ2);

// 	double xCoord = 0;
// 	double yCoord = 0;
// 	for (int i=0; i<num_node_Q1isoQ2; i++){
// 		if (xCoord-1 > 1e-6){
// 			xCoord = 0;
// 			yCoord += 1.0/division_Q1isoQ2;
// 		}
// 		points[0][i] = xCoord;
// 		points[1][i] = yCoord;
// 		xCoord += 1.0/division_Q1isoQ2;
// 	}

// 	quadrilaterals = std::make_unique<std::unique_ptr<int[]>[]>(4);
// 	for (int i=0; i<4; i++){
// 		quadrilaterals[i] = std::make_unique<int[]>(num_Q1isoQ2_element);
// 	}
// 	int refDof[4] = {0, 1, division_Q1isoQ2+2, division_Q1isoQ2+1};
// 	for (int i=0; i<num_Q1isoQ2_element; i++){
// 		quadrilaterals[0][i] = refDof[0];
// 		quadrilaterals[1][i] = refDof[1];
// 		quadrilaterals[2][i] = refDof[2];
// 		quadrilaterals[3][i] = refDof[3];

// 		if ((refDof[1]+1)%(division_Q1isoQ2+1) == 0){
// 			refDof[0] += 2;
// 			refDof[1] += 2;
// 			refDof[2] += 2;
// 			refDof[3] += 2;
// 		} else {
// 			refDof[0] += 1;
// 			refDof[1] += 1;
// 			refDof[2] += 1;
// 			refDof[3] += 1;
// 		}
// 	}

// 	Q1_idx = std::make_unique<int[]>(num_node_Q1isoQ2);
// 	for (int i = 0; i < num_node_Q1isoQ2; i++){
// 		Q1_idx[i] = num_node_Q1isoQ2;
// 	}

// 	int position = 0;
// 	int value = 0;
// 	for (int i = 0; i < division_Q1isoQ2/2.0+1; i++){
// 		position = 2*(division_Q1isoQ2+1)*i;
// 		for (int j = 0; j < division_Q1isoQ2/2.0+1; j++){
// 			Q1_idx[position] = value;
// 			value += 1; 
// 			position = position + 2;
// 		}
// 	}

// 	for (int i = 0; i < num_node_Q1isoQ2; i++){
// 		if (Q1_idx[i] == num_node_Q1isoQ2){
// 			Q1_idx[i] += i;
// 		}
// 	}

// 	std::vector<int> mesh_idx_vector(num_node_Q1isoQ2);
// 	std::vector<int> mesh_idx_vector2(num_node_Q1isoQ2);

// 	std::iota(mesh_idx_vector.begin(), mesh_idx_vector.end(), 0);
// 	std::iota(mesh_idx_vector2.begin(), mesh_idx_vector2.end(), 0);

// 	std::stable_sort(mesh_idx_vector.begin(), mesh_idx_vector.end(), [&](int i, int j){return Q1_idx[i] < Q1_idx[j];});
// 	std::stable_sort(mesh_idx_vector2.begin(), mesh_idx_vector2.end(), [&](int i, int j){return mesh_idx_vector[i] < mesh_idx_vector[j];});
// 	mesh_idx = std::make_unique<int[]>(num_node_Q1isoQ2);
// 	for (int i = 0; i < num_node_Q1isoQ2; i++){
// 		mesh_idx[i] = mesh_idx_vector2[i];
// 	}

// 	int size = division_Q1isoQ2*4;
// 	Bs = std::make_unique<int[]>(2*size);
// 	int b_pos[4] = {0, division_Q1isoQ2, num_node_Q1isoQ2-1, num_node_Q1isoQ2-1-division_Q1isoQ2};	 
// 	for (int i=0; i<division_Q1isoQ2; i++){
// 		Bs[i]                    = b_pos[0]+i;
// 		Bs[i+division_Q1isoQ2]   = b_pos[1]+(division_Q1isoQ2+1)*i;
// 		Bs[i+2*division_Q1isoQ2] = b_pos[2]-i;
// 		Bs[i+3*division_Q1isoQ2] = b_pos[3]-(division_Q1isoQ2+1)*i;
// 	}
// 	for (int i = 0; i < 4*division_Q1isoQ2; i++){
// 		Bs[i+4*division_Q1isoQ2] +=  Bs[i]+num_node_Q1isoQ2;
// 	}

// 	linear_system_setup();
// 	system_rhs();
// 	apply_boundary_condition();

// 	KSPCreate(PETSC_COMM_SELF, &ksp);
// 	KSPSetType(ksp, KSPFGMRES);
// 	// KSPGMRESSetRestart(ksp, 100);
// 	KSPSetOperators(ksp, finalMatrix, finalMatrix);
// 	KSPGetPC(ksp, &pc);
// 	PCSetType(pc, PCFIELDSPLIT);
// 	PCFieldSplitSetDetectSaddlePoint(pc, PETSC_TRUE);
// 	KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
// 	KSPMonitorSet(ksp, MyKSPMonitor, NULL, 0);
// 	KSPSetUp(ksp);

// 	u1 = std::make_unique<double[]>(num_node_Q1isoQ2);
// 	u2 = std::make_unique<double[]>(num_node_Q1isoQ2);
// };

// void stokesSolver::updateGeneratorSeed(double seed_){
// 	generator.seed(seed_);
// };

// void stokesSolver::apply_boundary_condition(){
// 	Vec workspace;
// 	VecDuplicate(states, &workspace);
// 	VecSet(workspace, 0.0);
// 	VecAssemblyBegin(workspace);
// 	VecAssemblyEnd(workspace);
// 	MatZeroRows(finalMatrix, 2*4*division_Q1isoQ2, Bs.get(), 1.0, workspace, finalRhs);
// 	// MatZeroRowsColumns(finalMatrix, 2*4*division_Q1isoQ2, Bs.get(), 1.0, states, finalRhs);
// };


// void stokesSolver::apply_homo_boundary_condition(){
// 	Vec workspace;
// 	VecDuplicate(states, &workspace);
// 	VecSet(workspace, 0.0);
// 	VecAssemblyBegin(workspace);
// 	VecAssemblyEnd(workspace);
// 	MatZeroRows(finalMatrix, 2*4*division_Q1isoQ2, Bs.get(), 1.0, workspace, adjointRhs);
// 	// MatZeroRowsColumns(finalMatrix, 2*4*division_Q1isoQ2, Bs.get(), 1.0, states, finalRhs);
// };


// void stokesSolver::linear_system_setup(){
// 	stiffness_matrix();
// 	B1_matrix();
// 	B2_matrix();
// 	system_matrix();
// 	Q1isoQ2Cond();	
// }

// void stokesSolver::forwardEqn(){
// 	system_rhs();
// 	apply_boundary_condition();

// 	KSPSolve(ksp, finalRhs, states);
// 	VecGetArray(states, &states_array);
// 	std::copy(states_array, states_array+num_node_Q1isoQ2, u1.get());
// 	std::copy(states_array+num_node_Q1isoQ2, states_array+2*num_node_Q1isoQ2, u2.get());
// 	// VecView(states, PETSC_VIEWER_STDOUT_WORLD);
// };

// void stokesSolver::pointObsMatrix(double obs[][4], int size){
// 	MatCreate(MPI_COMM_SELF, &state2obsMatrix);
// 	MatSetSizes(state2obsMatrix, PETSC_DECIDE, PETSC_DECIDE, 2*size, 2*num_node_Q1isoQ2+num_node_Q1);
// 	MatSetFromOptions(state2obsMatrix);
// 	MatSeqAIJSetPreallocation(state2obsMatrix, 6, NULL);
// 	MatMPIAIJSetPreallocation(state2obsMatrix, 6, NULL, 6, NULL);	

// 	double element_h = 1.0/division_Q1isoQ2;

// 	int column;
// 	int row;

// 	double local_x;
// 	double local_y;

// 	int square[4];
// 	double coords[4];

// 	double epsilon;
// 	double eta;

// 	for (int i = 0; i < size; i++){
// 		column  = obs[i][0] / element_h;
// 		row     = obs[i][1] / element_h;
// 		local_x = fmod(obs[i][0], element_h);
// 		local_y = fmod(obs[i][1], element_h);

// 		square[0] = quadrilaterals[0][division_Q1isoQ2*row+column];
// 		square[1] = quadrilaterals[1][division_Q1isoQ2*row+column];
// 		square[2] = quadrilaterals[2][division_Q1isoQ2*row+column];
// 		square[3] = quadrilaterals[3][division_Q1isoQ2*row+column];

// 		coords[0] =	points[0][square[0]];
// 		coords[1] =	points[0][square[2]];
// 		coords[2] =	points[1][square[0]];
// 		coords[3] =	points[1][square[2]];		

// 		epsilon = (obs[i][0]-coords[0])/(coords[1]-coords[0])*2-1;
// 		eta     = (obs[i][1]-coords[2])/(coords[3]-coords[2])*2-1;

// 		MatSetValue(state2obsMatrix, i, square[0], 0.25*(1.0-epsilon)*(1.0-eta), INSERT_VALUES);
// 		MatSetValue(state2obsMatrix, i, square[1], 0.25*(1.0+epsilon)*(1.0-eta), INSERT_VALUES);
// 		MatSetValue(state2obsMatrix, i, square[2], 0.25*(1.0+epsilon)*(1.0+eta), INSERT_VALUES);
// 		MatSetValue(state2obsMatrix, i, square[3], 0.25*(1.0-epsilon)*(1.0+eta), INSERT_VALUES);

// 		MatSetValue(state2obsMatrix, i+size, num_node_Q1isoQ2+square[0], 0.25*(1.0-epsilon)*(1.0-eta), INSERT_VALUES);
// 		MatSetValue(state2obsMatrix, i+size, num_node_Q1isoQ2+square[1], 0.25*(1.0+epsilon)*(1.0-eta), INSERT_VALUES);
// 		MatSetValue(state2obsMatrix, i+size, num_node_Q1isoQ2+square[2], 0.25*(1.0+epsilon)*(1.0+eta), INSERT_VALUES);
// 		MatSetValue(state2obsMatrix, i+size, num_node_Q1isoQ2+square[3], 0.25*(1.0-epsilon)*(1.0+eta), INSERT_VALUES);
//    	}
// 	MatAssemblyBegin(state2obsMatrix, MAT_FINAL_ASSEMBLY);
// 	MatAssemblyEnd(state2obsMatrix, MAT_FINAL_ASSEMBLY);
// 	// MatView(state2obsMatrix, PETSC_VIEWER_STDOUT_WORLD);
// }


// void stokesSolver::adjointEqn(int size, double obs[][4]){
// 	Vec workspace;
//  	VecCreate(MPI_COMM_SELF, &workspace);
// 	VecSetSizes(workspace, PETSC_DECIDE, 2*size);
// 	VecSetFromOptions(workspace);
// 	VecSet(workspace, 0.0);
// 	VecAssemblyBegin(workspace);
// 	VecAssemblyEnd(workspace);


//  	MatMult(state2obsMatrix, states, workspace);
// 	Vec workspace2;
//  	VecCreate(MPI_COMM_SELF, &workspace2);
// 	VecSetSizes(workspace2, PETSC_DECIDE, 2*size);
// 	VecSetFromOptions(workspace2);
//  	for (int i = 0; i < size; ++i){
//  		VecSetValue(workspace2, i, obs[i][2], INSERT_VALUES);
//  		VecSetValue(workspace2, size+i, obs[i][3], INSERT_VALUES);
//  	}
// 	VecAssemblyBegin(workspace2);
// 	VecAssemblyEnd(workspace2);
//  	VecAXPY(workspace, -1, workspace2);
//  	MatMultTranspose(state2obsMatrix, workspace, adjointRhs);

//  	VecDuplicate(adjointRhs, &adjoints);
//  	apply_homo_boundary_condition();
// 	KSPSolve(ksp, adjointRhs, adjoints);	
// }

// void stokesSolver::controlEqn(double grad[]){
// 	VecGetArray(adjoints, &adjoint_array);
// 	double workspace;
// 	double load[2*num_node_Q1isoQ2];
// 	std::cout << "gradient: ";
// 	for (int i = 0; i < 10; ++i){
// 		load_vector_i(i, load);
// 		workspace = 0;
// 		for (int j = 0; j < 2*num_node_Q1isoQ2; ++j){
// 			workspace += load[i]*adjoint_array[i];
// 		}
// 		grad[i] = u[i] - workspace;	
// 		std::cout << grad[i] << " ";
// 	}
// 	std::cout << std::endl;
// }

// void stokesSolver::WuuApply(Vec direction, int size){
// 	Vec workspace;
//  	VecCreate(MPI_COMM_SELF, &workspace);
// 	VecSetSizes(workspace, PETSC_DECIDE, 2*size);
// 	VecSetFromOptions(workspace);
// 	VecSet(workspace, 0.0);
// 	VecAssemblyBegin(workspace);
// 	VecAssemblyEnd(workspace);

// 	MatMult(state2obsMatrix, direction, workspace);
// 	MatMultTranspose(state2obsMatrix, workspace, direction);
// }

// Vec stokesSolver::CApply(Vec direction, int size, bool T){
// 	Mat C;
// 	MatCreate(MPI_COMM_SELF, &C);
// 	MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, 2*num_node_Q1isoQ2+num_node_Q1, 10);
// 	MatSetFromOptions(C);
// 	MatSeqAIJSetPreallocation(C, 10, NULL);
// 	MatMPIAIJSetPreallocation(C, 10, NULL, 10, NULL);	
// 	double load[2*num_node_Q1isoQ2];
// 	for (int i = 0; i < 10; ++i){
// 		load_vector_i(i, load);
// 		for (int j = 0; j < 2*num_node_Q1isoQ2+num_node_Q1; ++j){
// 			MatSetValue(C, j, i, load[j], INSERT_VALUES);
// 		}
// 	}
// 	MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
// 	MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);

// 	Vec workspace;
// 	if (T == 0){
// 		VecSetSizes(workspace, PETSC_DECIDE, 2*num_node_Q1isoQ2+num_node_Q1);
// 		VecSet(workspace, 0.0);
// 		VecAssemblyBegin(workspace);
// 		VecAssemblyEnd(workspace);
// 		MatMult(C, direction, workspace);
// 	} else {
// 		VecSetSizes(workspace, PETSC_DECIDE, 20);
// 		VecSet(workspace, 0.0);
// 		VecAssemblyBegin(workspace);
// 		VecAssemblyEnd(workspace);
// 		MatMultTranspose(C, direction, workspace);
// 	}
// 	return workspace;
// }

// Vec stokesSolver::ASolve(Vec direction, int size, bool T){
// 	Vec workspace;
// 	if (T == 0){
// 		KSPSolve(ksp, direction, workspace);
// 	} else {
// 		KSPSolveTranspose(ksp, direction, workspace);
// 	}
// 	return workspace;
// }

// Vec stokesSolver::RApply(Vec direction, bool T){
// 	Mat R;
// 	MatCreate(MPI_COMM_SELF, &R);
// 	MatSetSizes(R, PETSC_DECIDE, PETSC_DECIDE, 10, 10);
// 	MatSetFromOptions(R);
// 	MatSeqAIJSetPreallocation(R, 1, NULL);
// 	MatMPIAIJSetPreallocation(R, 1, NULL, 1, NULL);	
// 	for (int i = 0; i < 10; ++i){
// 		MatSetValue(R, i, i, 1.0, INSERT_VALUES);
// 	}
// 	MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY);
// 	MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY);

// 	Vec workspace;
// 	VecSetSizes(workspace, PETSC_DECIDE, 10);
// 	VecSet(workspace, 0.0);
// 	VecAssemblyBegin(workspace);
// 	VecAssemblyEnd(workspace);
// 	if (T == 0){
// 		MatMult(R, direction, workspace);
// 	} else {
// 		MatMultTranspose(R, direction, workspace);
// 	}

// 	return workspace;
// }

// void stokesSolver::HessianAction(double grad[]){
// 	Vec workspace;
// 	VecSetSizes(workspace, PETSC_DECIDE, 10);
// 	for (int i = 0; i < 10; ++i){
// 		VecSetValue(workspace, i, grad[i], INSERT_VALUES);
// 	}
// 	VecAssemblyBegin(workspace);
// 	VecAssemblyEnd(workspace);	

// 	Vec workspace2;
// 	workspace2 = RApply(workspace, false);

// 	Vec workspace3;
// 	Vec workspace4;
// 	workspace3 = CApply(workspace, 10, false);
// 	workspace4 = ASolve(workspace3, 10, false);
// 	WuuApply(workspace4, 10);
// 	workspace3 = ASolve(workspace4, 10, true);
// 	workspace = CApply(workspace4, 10, true);

// 	VecAXPY(workspace, 1, workspace4);
// 	VecGetArray(workspace, &grad);
// }

// void stokesSolver::fullHessian(){
// 	double hessian[num_term][num_term];
// 	double input[num_term];
// 	for (int i = 0; i < num_term; ++i){
// 		for (int j = 0; j < num_term; ++j){
// 			if (i == j){
// 				input[j] = 1;
// 			} else {
// 				input[j] = 0;
// 			}
// 		}
// 		HessianAction(input);
// 		for (int j = 0; j< num_term; ++j){
// 			hessian[j][i] = input[j];
// 			std::cout << input[j] << " ";
// 		}
// 		std::cout << std::endl;
// 	}
// }

// // void stokesSolver::ReducedHessian(){
// // 	// std::cout << "create random matrix" << std::endl;
// // 	double randomOmega[num_term][15];
// // 	for (unsigned i = 0; i < num_term; i++){
// // 		for (unsigned j = 0; j < 15; j++){
// // 			randomOmega[i][j] = distribution(generator);
// // 		}
// // 	}	

// // 	double inputSpace[num_term];
// // 	double outputSpace[num_term];
// // 	MatCreateSeqDense(geo->comm, num_term, 15, NULL, &yMat);
// // 	for (int i = 0; i < 15; i++){
// // 		for (int j = 0; j < num_term; ++j){
// // 			inputSpace[j] = randomOmega[j][i];
// // 		}
// // 		HessianAction(inputSpace);

// // 		for (int j = 0; j < num_term; j++){
// // 			MatSetValue(yMat, j, i, inputSpace[j], INSERT_VALUES);
// // 		}
// // 	}
// // 	MatAssemblyBegin(yMat, MAT_FINAL_ASSEMBLY);
// // 	MatAssemblyEnd(yMat, MAT_FINAL_ASSEMBLY);

// // 	// std::cout << "yMat size" << std::endl;
// // 	// MatView(yMat, PETSC_VIEWER_STDOUT_WORLD);

// // 	// std::cout << "QR decomposition" << std::endl;

// // 	BV bv;
// // 	Mat Q;
// // 	BVCreateFromMat(yMat, &bv);
// // 	BVSetFromOptions(bv);
// // 	BVOrthogonalize(bv, NULL);
// // 	BVCreateMat(bv, &Q);

// // 	// std::cout << "initialized eigenvalue problem" << std::endl;

// // 	EPS eps;
// // 	Mat T, S, lambda, U, UTranspose, ReducedHessian;
// // 	Vec workSpace1, workspace2, xr, xi;
// // 	PetscScalar kr, ki;
// // 	MatCreate(geo->comm, &T);
// // 	MatCreate(geo->comm, &S);
// // 	MatCreate(geo->comm, &U);
// // 	MatCreate(geo->comm, &lambda);
// // 	VecCreate(geo->comm, &workSpace1);
// // 	MatSetSizes(T, PETSC_DECIDE, PETSC_DECIDE, 15, 15);
// // 	MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, 15, 15);
// // 	MatSetSizes(U, PETSC_DECIDE, PETSC_DECIDE, 15, 15);
// // 	MatSetSizes(lambda, PETSC_DECIDE, PETSC_DECIDE, 15, 15);
// // 	VecSetSizes(workSpace1, PETSC_DECIDE, 15);
// // 	MatSetFromOptions(T);
// // 	MatSetFromOptions(S);
// // 	MatSetFromOptions(U);
// // 	MatSetFromOptions(lambda);
// // 	VecSetFromOptions(workSpace1);
// // 	MatSetUp(T);
// // 	MatSetUp(S);
// // 	MatSetUp(lambda);
// // 	MatSetUp(U);
// // 	VecSetUp(workSpace1);

// // 	// std::cout << "Assemble T matrix" << std::endl;
// // 	for (int i = 0; i < 15; i++){
// // 		MatGetColumnVector(Q, workspace1, i);
// // 		VecGetArray(workspace1, &inputSpace);

// // 		HessianAction(inputSpace);
// // 		for (int j = 0; j < num_term; ++j){
// // 			VecSetValue(workspace1, j, inputSpace[j]);		
// // 		}
// // 		VecAssemblyBegin(workspace1);
// // 		VecAssemblyEnd(workspace1);

// // 		MatMultTranspose(Q, workspace1, workSpace2);
// // 		VecGetArray(workSpace2, &outputSpace);
// // 		for (int j = 0; j < 15; j++){
// // 			MatSetValue(T, j, i, outputSpace[j], INSERT_VALUES);
// // 		}
// // 	}
// // 	MatAssemblyBegin(T, MAT_FINAL_ASSEMBLY);
// // 	MatAssemblyEnd(T, MAT_FINAL_ASSEMBLY);
// // 	MatCreateVecs(T, NULL, &xr);
// // 	MatCreateVecs(T, NULL, &xi);

// // 	// std::cout << "start EPS solver" << std::endl;

// // 	EPSCreate(PETSC_COMM_SELF, &eps);
// // 	EPSSetOperators(eps, T, NULL);
// // 	EPSSetFromOptions(eps);
// // 	EPSSolve(eps);

// // 	PetscScalar *eigenValues;
// // 	for (int i = 0; i < 15; i++){
// // 		EPSGetEigenpair(eps, i, &kr, &ki, xr, xi);
// // 		VecGetArray(xr, &eigenValues);
// // 		for (int j = 0; j < 15; j++){
// // 			if (i == j){
// // 				MatSetValue(lambda, j, i, kr, INSERT_VALUES);
// // 			} else {
// // 				MatSetValue(lambda, j, i, 0, INSERT_VALUES);
// // 			}
// // 			MatSetValue(S, j, i, eigenValues[j], INSERT_VALUES);
// // 		}
// // 	}

// // 	MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
// // 	MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);
// // 	MatAssemblyBegin(lambda, MAT_FINAL_ASSEMBLY);
// // 	MatAssemblyEnd(lambda, MAT_FINAL_ASSEMBLY);

// // 	std::cout << "compute hessian matrix" << std::endl;

// // 	MatMatMult(Q, S, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &U);
// // 	MatTranspose(U, MAT_INITIAL_MATRIX, &UTranspose);

// // 	// MatPtAP(lambda, UTranspose, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &ReducedHessian);
// // 	// MatView(ReducedHessian, PETSC_VIEWER_STDOUT_WORLD);

// // 	// std::cout << "convert PETSc to Dolfin" << std::endl;
// // 	// std::shared_ptr<Matrix> out(new Matrix(PETScMatrix(ReducedHessian)));

// // 	// std::cout << "Mat Destroy Reduced Hessian" << std::endl;
// // 	// MatDestroy(&ReducedHessian);

// // 	// std::cout << "hessianMatrix" << std::endl;
// // }

// double stokesSolver::solve(double obs[][4], int size)
// {
// 	forwardEqn();
// 	return lnlikelihood(obs, size);
// }

// void stokesSolver::pCNProposalGenerator(double proposal[])
// {
// 	for (int m_idx = 0; m_idx < num_term; ++m_idx){
// 		proposal[m_idx] = sqrt(1-beta*beta)*proposal[m_idx] + beta*distribution(generator);
// 	}
// }

// void stokesSolver::SNProposalGenerator(double proposal[]){
	
// }

// void stokesSolver::getValues(double obs[][2], double velocity[][2], int size)
// {
// 	double element_h = 1.0/division_Q1isoQ2;

// 	int column;
// 	int row;

// 	double local_x;
// 	double local_y;

// 	int square[4];
// 	double coords[4];

// 	double epsilon;
// 	double eta;

// 	double point_u1;
// 	double point_u2;

// 	double local_u1[4];
// 	double local_u2[4];

// 	for (int i = 0; i < size; ++i){
// 		column  = obs[i][0] / element_h;
// 		row     = obs[i][1] / element_h;
// 		local_x = fmod(obs[i][0], element_h);
// 		local_y = fmod(obs[i][1], element_h);

// 		if (column == division_Q1isoQ2){
// 			column -= 1;
// 		}
// 		if (row == division_Q1isoQ2){
// 			row -= 1;
// 		}

// 		square[0] = quadrilaterals[0][division_Q1isoQ2*row+column];
// 		square[1] = quadrilaterals[1][division_Q1isoQ2*row+column];
// 		square[2] = quadrilaterals[2][division_Q1isoQ2*row+column];
// 		square[3] = quadrilaterals[3][division_Q1isoQ2*row+column];

// 		coords[0] =	points[0][square[0]];
// 		coords[1] =	points[0][square[2]];
// 		coords[2] =	points[1][square[0]];
// 		coords[3] =	points[1][square[2]];		

// 		epsilon = (obs[i][0]-coords[0])/(coords[1]-coords[0])*2-1;
// 		eta     = (obs[i][1]-coords[2])/(coords[3]-coords[2])*2-1;

// 		for (int k = 0; k < 4; ++k){
// 			local_u1[k] = u1[square[k]];
// 			local_u2[k] = u2[square[k]];
// 		}

// 		velocity[i][0] = shape_interpolation(epsilon, eta, local_u1);
// 		velocity[i][1] = shape_interpolation(epsilon, eta, local_u2);
// 	}
// }

// double stokesSolver::lnlikelihood(double obs[][4], int size)
// {
// 	double element_h = 1.0/division_Q1isoQ2;

// 	int column;
// 	int row;

// 	double local_x;
// 	double local_y;

// 	int square[4];
// 	double coords[4];

// 	double epsilon;
// 	double eta;

// 	double point_u1;
// 	double point_u2;

// 	double local_u1[4];
// 	double local_u2[4];

// 	double misfit = 0;

// 	for (int i = 0; i < size; i++){
// 		column  = obs[i][0] / element_h;
// 		row     = obs[i][1] / element_h;
// 		local_x = fmod(obs[i][0], element_h);
// 		local_y = fmod(obs[i][1], element_h);

// 		square[0] = quadrilaterals[0][division_Q1isoQ2*row+column];
// 		square[1] = quadrilaterals[1][division_Q1isoQ2*row+column];
// 		square[2] = quadrilaterals[2][division_Q1isoQ2*row+column];
// 		square[3] = quadrilaterals[3][division_Q1isoQ2*row+column];

// 		coords[0] =	points[0][square[0]];
// 		coords[1] =	points[0][square[2]];
// 		coords[2] =	points[1][square[0]];
// 		coords[3] =	points[1][square[2]];		

// 		epsilon = (obs[i][0]-coords[0])/(coords[1]-coords[0])*2-1;
// 		eta     = (obs[i][1]-coords[2])/(coords[3]-coords[2])*2-1;

// 		for (int k = 0; k < 4; ++k){
// 			local_u1[k] = u1[square[k]];
// 			local_u2[k] = u2[square[k]];
// 		}

// 		point_u1 = shape_interpolation(epsilon, eta, local_u1);
// 		point_u2 = shape_interpolation(epsilon, eta, local_u2);
      
//       	//std::cout << "point u: " << point_u1 << " " << point_u2 << " obs: " << obs[i][2] << " " << obs[i][3] << std::endl; 
//         misfit += std::pow(point_u1-obs[i][2], 2.) + std::pow(point_u2-obs[i][3], 2.);
// 	}
// 	return -misfit/noiseVariance;
// }

// void obs_gen(int level, std::vector<double> &rand_coef, double obs[][4], int obsNumPerRow, double noiseVariance)
// {
// 	stokesSolver obsSolver(level, rand_coef.size(), noiseVariance);
// 	for (int i = 0; i < rand_coef.size(); ++i){
// 		obsSolver.u[i] = rand_coef[i];
// 	}
// 	obsSolver.forwardEqn();

// 	int obsIdx[obsNumPerRow*obsNumPerRow];
// 	double incremental = (1./obsNumPerRow)/(1./obsSolver.division_Q1isoQ2);
//     double Idx1 = (1./2./obsNumPerRow)/(1./obsSolver.division_Q1isoQ2) + (1./2./obsNumPerRow)/(1./obsSolver.division_Q1isoQ2)*(obsSolver.division_Q1isoQ2+1.);

//     for (int i = 0; i < obsNumPerRow; i++){  
//     	obsIdx[obsNumPerRow*i] = Idx1 + i*incremental*(obsSolver.division_Q1isoQ2+1);
//     }

// 	for (int i = 0; i < obsNumPerRow; i++){
// 		for (int j = 0; j < obsNumPerRow; j++){
// 			obsIdx[obsNumPerRow*i+j] = obsIdx[obsNumPerRow*i]+j*incremental;   		
// 		}
// 	}    
 
// 	for (int i = 0; i < obsNumPerRow*obsNumPerRow; i++){
// 		obs[i][0] = obsSolver.points[0][obsIdx[i]]; 
// 		obs[i][1] = obsSolver.points[1][obsIdx[i]];
// 		obs[i][2] = obsSolver.u1[obsIdx[i]];
// 		obs[i][3] = obsSolver.u2[obsIdx[i]];
// 	}
// }

// void plain_mcmc(stokesSolver *solver, int max_samples, double obs[][4], int obs_size, double mean[], int coef_size)
// {
//     std::uniform_real_distribution<double> uni_dist(0.0, 1.0);
//     double u[coef_size];
//     double u_next[coef_size];
//     double sum[coef_size];
//     double uniform_alpha;
//     double alpha;

//     for (int i = 0; i < coef_size; ++i){
//     	solver->u[i] = solver->distribution(solver->generator);
//     	u[i] = solver->u[i];
//     	u_next[i] = solver->u[i];
//     	sum[i] = solver->u[i];
//     }

//     double lnlikelihood;
//     double lnlikelihood_next;
//     lnlikelihood = solver->solve(obs, obs_size);

//     for (int i = 0; i < max_samples; i++){
//         solver->pCNProposalGenerator(u_next);
//         for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//             solver->u[m_idx] = u_next[m_idx];
//         }        

// 	lnlikelihood_next = solver->solve(obs, obs_size);

//         alpha = std::min(0.0, lnlikelihood_next - lnlikelihood);
//         uniform_alpha = std::log(uni_dist(solver->generator));

//         if (uniform_alpha < alpha){
//             std::cout << "accepted: " << uniform_alpha << ", " << alpha << std::endl; 
//             for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//                 u[m_idx] = u_next[m_idx];
//             }
//             lnlikelihood = lnlikelihood_next;
//         } else {         
// 	    std::cout << "rejected: " << uniform_alpha << ", " << alpha << std::endl; 
//             for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//                 u_next[m_idx] = u[m_idx];
//             }	
//         }
//         for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//             sum[m_idx] += u[m_idx];
//              std::cout << u[m_idx] << " ";
//         }
//         //std::cout << std::endl;
//         // if (i % 10 == 0){
//         // 	std::cout << "average: " << sum[0]/(i+1) << " " << sum[1]/(i+1) << " " << sum[2]/(i+2) << std::endl;
//         // }
//     }
//     //std::cout << "final mean: " << std::endl;
//     for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//         mean[m_idx] = sum[m_idx] / (max_samples+1);
//         //std::cout << mean[m_idx] << " ";
//     }
// }

// void ml_mcmc(int levels, std::vector<std::shared_ptr<stokesSolver>> solvers, double obs[][4], double out[], int size, int a, int coef_size)
// {
//     std::uniform_real_distribution<double> uni_dist(0.0, 1.0);
//     std::normal_distribution<double> norm_dist(0.0, 1.0);

//     // int M = pow(levels, a-2)*pow(2.0, 2.0*levels);
//     int M = levels*pow(2.0, 2.0*levels);
//     double mean[coef_size];
//     plain_mcmc(solvers[0].get(), M, obs, size, mean, coef_size);
//     double L[levels+1][coef_size];
//     std::cout << "L0: ";
//     for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//         L[0][m_idx] = mean[m_idx];
//         std::cout << mean[m_idx] << " ";
//     }
//     std::cout << std::endl;

//     double u[coef_size];
//     double u_next[coef_size];
//     double lnlikelihoodUpper;
//     double lnlikelihoodLower;
//     double lnlikelihoodUpper_next;
//     double lnlikelihoodLower_next;
//     double alpha;
//     double uniform_alpha;
//     int acceptance;
//     int I;

//     double A0[coef_size];
//     double A1[coef_size];
//     double A2[coef_size];
//     double A3[coef_size];
//     double A4[coef_size];
//     double A5[coef_size];
//     double A6[coef_size];
//     double A7[coef_size];

//     double A0Sum[coef_size];
//     double A1Sum[coef_size];
//     double A2Sum[coef_size];
//     double A3Sum[coef_size];
//     double A4Sum[coef_size];
//     double A5Sum[coef_size];
//     double A6Sum[coef_size];
//     double A7Sum[coef_size];

//     for (int i = 0; i < levels; i++){
//         M = pow(i+1, 3)*pow(2.0,2.0*(levels-i-1.0));
// 	    for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//                 A0Sum[m_idx] = 0;
//                 A1Sum[m_idx] = 0;
//                 A2Sum[m_idx] = 0;
//                 A3Sum[m_idx] = 0;
//                 A4Sum[m_idx] = 0;
//                 A5Sum[m_idx] = 0;
//                 A6Sum[m_idx] = 0;
//                 A7Sum[m_idx] = 0;
//             }

//     	for (int chainNum = 0; chainNum < 2; chainNum++){
//             std::shared_ptr<stokesSolver> solverUpper = solvers[i+1];
//             std::shared_ptr<stokesSolver> solverLower = solvers[i];
            
//             for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//                 u[m_idx] = norm_dist(solverLower->generator);
//                 u_next[m_idx] = u[m_idx];
//                 solverUpper->u[m_idx] = u[m_idx];
//                 solverLower->u[m_idx] = u[m_idx];
//             }

//             if (chainNum == 0){
//                 lnlikelihoodUpper = solverUpper->solve(obs, size);
//                 lnlikelihoodLower = solverLower->solve(obs, size);
//             } else {
//                 lnlikelihoodLower = solverLower->solve(obs, size);
//                 lnlikelihoodUpper = solverUpper->solve(obs, size);
//             }
//             if (-lnlikelihoodUpper+lnlikelihoodLower < 0){
//                 I = 1;
//             } else {
//                 I = 0;   
//             }

//             if (chainNum == 0){
//                 if (I == 0){
//                     for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//                         A0[m_idx] = 0;
//                         A2[m_idx] = 0;
//                         A5[m_idx] = 0;
//                         A6[m_idx] = u[m_idx]*(1-I);
//                         A0Sum[m_idx] += A0[m_idx];
//                         A2Sum[m_idx] += A2[m_idx];
//                         A5Sum[m_idx] += A5[m_idx];
//                         A6Sum[m_idx] += A6[m_idx];
//                     }
//                 } else {
//                     for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//                         A0[m_idx] = (1-exp(-lnlikelihoodUpper+lnlikelihoodLower))*u[m_idx]*I;
//                         A2[m_idx] = (exp(-lnlikelihoodUpper+lnlikelihoodLower)-1)*I;
//                         A5[m_idx] = exp(-lnlikelihoodUpper+lnlikelihoodLower)*u[m_idx]*I;
//                         A6[m_idx] = 0;
//                         A0Sum[m_idx] += A0[m_idx]; 
//                         A2Sum[m_idx] += A2[m_idx]; 
//                         A5Sum[m_idx] += A5[m_idx]; 
//                         A6Sum[m_idx] += A6[m_idx]; 
//             	    }
//                 }
//             } else {
//                 if (I == 1){
//                     for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//                         A1[m_idx] = 0; 
//                         A3[m_idx] = u[m_idx]; 
//                         A4[m_idx] = 0; 
//                         A7[m_idx] = 0; 
//                         A1Sum[m_idx] += A1[m_idx]; 
//                         A3Sum[m_idx] += A3[m_idx]; 
//                         A4Sum[m_idx] += A4[m_idx]; 
//                         A7Sum[m_idx] += A7[m_idx];
//                     }                  
//                 } else {
//                     for (int m_idx = 0; m_idx < coef_size; ++m_idx){
// 	                    A1[m_idx] = (exp(-lnlikelihoodLower+lnlikelihoodUpper)-1)*u[m_idx]*(1-I);
// 	                    A3[m_idx] = 0;
// 	                    A4[m_idx] = (1-exp(-lnlikelihoodLower+lnlikelihoodUpper))*(1-I);
// 	                    A7[m_idx] = exp(-lnlikelihoodLower+lnlikelihoodUpper)*u[m_idx]*(1-I);
// 	                    A1Sum[m_idx] += A1[m_idx];
// 	                    A3Sum[m_idx] += A3[m_idx];
// 	                    A4Sum[m_idx] += A4[m_idx];
// 	                    A7Sum[m_idx] += A7[m_idx];     
//             	    }
//                  }
//             }
//             for (int sampleNum = 0; sampleNum < M; sampleNum++){
//                 if (chainNum == 0){
//                     solverUpper->pCNProposalGenerator(u_next);
//                     for (int m_idx = 0; m_idx < coef_size; ++m_idx){
// 	                    solverUpper->u[m_idx] = u_next[m_idx];
// 	                    solverLower->u[m_idx] = u_next[m_idx];   
//                     }
//                     lnlikelihoodUpper_next = solverUpper->solve(obs, size);
//                     alpha = std::min(0.0, lnlikelihoodUpper_next - lnlikelihoodUpper);
//                     uniform_alpha = log(uni_dist(solverUpper->generator));
//                     if (uniform_alpha < alpha){
//                         acceptance = 1;
//                     } else {
//                         acceptance = 0;
//                     }
//                 } else {
//                     solverLower->pCNProposalGenerator(u_next);
//                     for (int m_idx = 0; m_idx < coef_size; ++m_idx){
// 	                    solverUpper->u[m_idx] = u_next[m_idx];
// 	                    solverLower->u[m_idx] = u_next[m_idx];
//                     }
//                     lnlikelihoodLower_next = solverLower->solve(obs, size);
//                     alpha = std::min(0.0, lnlikelihoodLower_next - lnlikelihoodLower);
//                     uniform_alpha = log(uni_dist(solverLower->generator));
//                     if (uniform_alpha < alpha){
//                         acceptance = 1;
//                     } else {
//                         acceptance = 0;
//                     }
//                 }

//                 if (acceptance == 1){
//                 	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//                 		u[m_idx] = u_next[m_idx];
//                 	}

//                     if (chainNum == 0) {
//                         lnlikelihoodUpper = lnlikelihoodUpper_next;
//                         lnlikelihoodLower = solverLower->solve(obs, size);
//                     } else {
//                         lnlikelihoodUpper = solverUpper->solve(obs, size);
//                         lnlikelihoodLower = lnlikelihoodLower_next;
//                     }
//                     if (-lnlikelihoodUpper+lnlikelihoodLower < 0){
//                         I = 1;
//                     } else {
//                         I = 0; 
//                     }

//                     if (chainNum == 0){
//                         if (I == 0){
//                         	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
// 	                            A0[m_idx] = 0;
// 	                            A2[m_idx] = 0;
// 	                            A5[m_idx] = 0;
// 	                            A6[m_idx] = u[m_idx]*(1-I);
// 			                    A0Sum[m_idx] += A0[m_idx]; 
// 			                    A2Sum[m_idx] += A2[m_idx]; 
// 			                    A5Sum[m_idx] += A5[m_idx]; 
// 			                    A6Sum[m_idx] += A6[m_idx];  

//                         	}                          
//                         } else {
//                         	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
// 	                		    A0[m_idx] = (1-exp(-lnlikelihoodUpper+lnlikelihoodLower))*u[m_idx]*I;
// 	                    		A2[m_idx] = (exp(-lnlikelihoodUpper+lnlikelihoodLower)-1)*I;
// 	                    		A5[m_idx] = exp(-lnlikelihoodUpper+lnlikelihoodLower)*u[m_idx]*I;
// 	                    		A6[m_idx] = 0; 
// 			                    A0Sum[m_idx] += A0[m_idx];
// 			                    A2Sum[m_idx] += A2[m_idx];
// 			                    A5Sum[m_idx] += A5[m_idx];
// 			                    A6Sum[m_idx] += A6[m_idx];
//                         	}
//                     	}
//                     } else {
//                         if (I == 1){
//                         	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
// 			                    A1[m_idx] = 0; 
// 			                    A3[m_idx] = u[m_idx]; 
// 			                    A4[m_idx] = 0; 
// 			                    A7[m_idx] = 0; 
// 			                    A1Sum[m_idx] += A1[m_idx]; 
// 			                    A3Sum[m_idx] += A3[m_idx]; 
// 			                    A4Sum[m_idx] += A4[m_idx]; 
// 			                    A7Sum[m_idx] += A7[m_idx];   
//                         	}        
//                         } else {
//                         	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
// 			                    A1[m_idx] = (exp(-lnlikelihoodLower+lnlikelihoodUpper)-1)*u[m_idx]*(1-I);
// 			                    A3[m_idx] = 0;
// 			                    A4[m_idx] = (1-exp(-lnlikelihoodLower+lnlikelihoodUpper))*(1-I);
// 			                    A7[m_idx] = exp(-lnlikelihoodLower+lnlikelihoodUpper)*u[m_idx]*(1-I);
// 			                    A1Sum[m_idx] += A1[m_idx];
// 			                    A3Sum[m_idx] += A3[m_idx];
// 			                    A4Sum[m_idx] += A4[m_idx];
// 			                    A7Sum[m_idx] += A7[m_idx];  
//                         	}
//                 		}
//                 	}
//                 } else {
//                 	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//                 		solverUpper->u[m_idx] = u[m_idx];
//                 		solverLower->u[m_idx] = u[m_idx];
//                 	}
//                     if (chainNum == 0){
//                     	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
// 		                    A0Sum[m_idx] += A0[m_idx]; 
// 		                    A2Sum[m_idx] += A2[m_idx]; 
// 		                    A5Sum[m_idx] += A5[m_idx]; 
// 		                    A6Sum[m_idx] += A6[m_idx];    
//                     	}
  
//                     } else {
//                     	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
// 		                    A1Sum[m_idx] += A1[m_idx];
// 		                    A3Sum[m_idx] += A3[m_idx];
// 		                    A4Sum[m_idx] += A4[m_idx];
// 		                    A7Sum[m_idx] += A7[m_idx];  
//                     	}
//     				}
//     			}
//     			//std::cout << u[0] << " " << u[1] << " " << u[2] << " " << u[3] << std::endl;
//     			if (chainNum == 0){
// 	    			std::cout << "u value: " << u[0] << " likelihood: " << lnlikelihoodUpper << " " << lnlikelihoodLower  << " I: " << I << " A0: " << A0[0] << " A2: " << A2[0] << " A5: " << A5[0] << " A6: " << A6[0] << std::endl;
//     			} else {
// 	    			std::cout << "u value: " << u[0] << " likelihood: " << lnlikelihoodUpper << " " << lnlikelihoodLower  << " I: " << I << " A1: " << A1[0] << " A3: " << A3[0] << " A4: " << A4[0] << " A7: " << A7[0] << std::endl;    				
//     			}
//     		}
//     	}	
//     	std::cout << "L" << i+1 << ": ";
//     	for (int m_idx = 0; m_idx < coef_size; ++m_idx){
// 	        L[i+1][m_idx] = A0Sum[m_idx]/(M+1)+A1Sum[m_idx]/(M+1)+A2Sum[m_idx]/(M+1)*(A3Sum[m_idx]/(M+1)+A7Sum[m_idx]/(M+1))+A4Sum[m_idx]/(M+1)*(A5Sum[m_idx]/(M+1)+A6Sum[m_idx]/(M+1));
//     		std::cout << L[i+1][m_idx] << " ";
//     	}
//     	std::cout << std::endl;
//     }
//     for (int m_idx = 0; m_idx < coef_size; ++m_idx){
//     	out[m_idx] = 0.0;
//     }
//     for (int i = 0; i < levels; i++){
// 	    for (int m_idx = 0; m_idx < coef_size; ++m_idx){
// 	    	out[m_idx] += L[i][m_idx];
// 	    }
//     }
// }

// void write2txt(double array[][4], int arraySize, std::string pathName){
// 	std::ofstream myfile;
// 	myfile.open(pathName);
// 	for (int j = 0; j < 4; ++j){
// 		for (int i = 0; i < arraySize; ++i){
// 			myfile << array[i][j] << " ";
// 		}
// 		myfile << std::endl;
// 	}
// 	myfile.close();
// };


// void writeU(double u1[], double u2[], int arraySize, std::string pathName){
// 	std::ofstream myfile;
// 	myfile.open(pathName);
// 	for (int i = 0; i < arraySize; ++i){
// 		myfile << u1[i] << " " << u2[i] << std::endl;
// 	}
// 	myfile.close();
// };


// void txt2read(double array[][4], int arraySize, std::string pathName){
// 	std::ifstream myfile;
// 	myfile.open(pathName, std::ios_base::in);
// 	for (int j = 0; j < 4; ++j){
// 		for(int i = 0; i < arraySize; ++i){
// 			myfile >> array[i][j];
// 		}
// 	}
// 	myfile.close();
// }

// void read_config(std::vector<std::string> &paras, std::vector<std::string> &vals){
// 	std::ifstream cFile("mlmcmc_config.txt");
// 	if (cFile.is_open()){
// 		std::string line;
// 		while(getline(cFile, line)){
// 			line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
// 			if (line[0] == '#' || line.empty()){
// 				continue;
// 			}
// 			auto delimiterPos = line.find("=");
// 			auto name = line.substr(0, delimiterPos);
// 			auto value = line.substr(delimiterPos+1);

// 			paras.push_back(name);
// 			vals.push_back(value);
// 		}
// 	}
// 	cFile.close();
// }

// int main(int argc, char* argv[]){
// 	int rank;
// 	int size;

// 	PetscInitialize(&argc, &argv, NULL, NULL);
// 	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
// 	MPI_Comm_size(PETSC_COMM_WORLD, &size);

// 	std::vector<std::string> name;
// 	std::vector<std::string> value;
// 	std::vector<double> rand_coef;
// 	read_config(name, value);
// 	int num_term = std::stoi(value[0]);
// 	for (int i = 0; i < num_term; ++i){
// 		rand_coef.push_back(std::stod(value[i+1]));
// 	}
// 	int levels = std::stoi(value[num_term+1]);
// 	int a = std::stoi(value[num_term+2]);
// 	double pCN = std::stod(value[num_term+3]);
// 	int task = std::stoi(value[num_term+4]);
// 	int plainMCMC_sample_number = std::stoi(value[num_term+5]);
// 	int obsNumPerRow = std::stoi(value[num_term+6]);
// 	double noiseVariance = std::stod(value[num_term+7]);

// 	if (rank == 0){
// 		std::cout << "configuration: " << std::endl; 
// 		std::cout << "num_term: " << num_term << " coefs: ";
// 		for (int i = 0; i < num_term; ++i){
// 			std::cout << rand_coef[i] << " ";
// 		}
// 		std::cout << std::endl;
// 		std::cout << "levels: "       << levels       <<  std::endl;
// 		std::cout << "a: "            << a            <<  std::endl;
// 		std::cout << "pCN: "          << pCN          <<  std::endl;
// 		std::cout << "task: "         << task         <<  std::endl;
// 		std::cout << "plainMCMC samples:"<< plainMCMC_sample_number << std::endl;
// 		std::cout << "obsNumPerRow: " << obsNumPerRow <<  std::endl;
// 		std::cout << "noiseVariance: " << noiseVariance << std::endl; 
// 	}
// 	//Generation of Observation
// 	double obs[obsNumPerRow*obsNumPerRow][4];
// 	std::string obs_file = std::to_string(levels+1);
// 	obs_file.append("_obs_para_");
// 	obs_file.append(std::to_string(num_term));
// 	obs_file.append("_");
// 	obs_file.append(std::to_string(obsNumPerRow));
// 	obs_file.append("_wNoise_");
// 	obs_file.append(std::to_string(noiseVariance));

// 	if (task == 0){ //Generate Observation
// 		obs_gen(levels+1, rand_coef, obs, obsNumPerRow, noiseVariance);

// 		std::normal_distribution<double> distribution{0.0, sqrt(noiseVariance)};
// 		std::default_random_engine generator;
// 		for (int i = 0; i < obsNumPerRow*obsNumPerRow; ++i){
// 			obs[i][2] = obs[i][2] + distribution(generator);
// 			obs[i][3] = obs[i][3] + distribution(generator);
// 		}
// 		write2txt(obs, obsNumPerRow*obsNumPerRow, obs_file);
// 	}

// 	if (task == 1){ //Run plain MCMC
// 		txt2read(obs, obsNumPerRow*obsNumPerRow, obs_file);
// 		stokesSolver* samplerSolver = new stokesSolver(levels, num_term, noiseVariance);
// 		samplerSolver->updateGeneratorSeed(rank*13.0);
// 		double mean[num_term];
// 		plain_mcmc(samplerSolver, plainMCMC_sample_number, obs, obsNumPerRow*obsNumPerRow, mean, num_term);
// 		delete samplerSolver;

// 		std::string outputfile = "plain_mcmc_output_";
// 		outputfile.append(std::to_string(rank));

// 		std::ofstream myfile;
// 		myfile.open(outputfile);
// 		for (int i = 0; i<num_term; ++i){
// 			myfile << mean[i] << " ";
// 		}
// 		myfile << std::endl;
// 		myfile.close();

// 		MPI_Barrier(MPI_COMM_WORLD);
// 		if (rank == 0){
// 			double buffer;
// 			std::string finaloutput = "plainmcmc_finalOutput";
// 			std::ofstream outputfile;
// 			outputfile.open(finaloutput);
// 			for (int i = 0; i < size; i++){
// 				std::string finalinput = "plain_mcmc_output_";
// 				finalinput.append(std::to_string(i));
// 				std::ifstream inputfile;
// 				inputfile.open(finalinput, std::ios_base::in);
// 				for(int i = 0; i < num_term; ++i){
// 					inputfile >> buffer;
// 					outputfile << buffer << " ";
// 				}
// 				outputfile << std::endl;
// 				inputfile.close();
// 			}
// 			outputfile.close();
// 		}
// 	}

// 	if (task == 2){ //Run MLMCMC
// 		txt2read(obs, obsNumPerRow*obsNumPerRow, obs_file);
// 		std::vector<std::shared_ptr<stokesSolver>> solvers(levels+1);
// 		for (int i = 0; i < levels+1; i++){
// 			solvers[i] = std::make_shared<stokesSolver>(i, num_term, noiseVariance);
// 			solvers[i]->beta = pCN;
// 			solvers[i]->updateGeneratorSeed(rank*13.0);
// 		} 
// 		double out[num_term];
// 		ml_mcmc(levels, solvers, obs, out, obsNumPerRow*obsNumPerRow, a, num_term);

// 		std::string outputfile = "output_";
// 		outputfile.append(std::to_string(rank));

// 		std::ofstream myfile;
// 		myfile.open(outputfile);
// 		for (int i = 0; i<num_term; ++i){
// 			myfile << out[i] << " ";
// 		}
// 		myfile << std::endl;
// 		myfile.close();

// 		MPI_Barrier(MPI_COMM_WORLD);
// 		if (rank == 0){
// 			double buffer;
// 			std::string finaloutput = "finalOutput";
// 			std::ofstream outputfile;
// 			outputfile.open(finaloutput);
// 			for (int i = 0; i < size; i++){
// 				std::string finalinput = "output_";
// 				finalinput.append(std::to_string(i));
// 				std::ifstream inputfile;
// 				inputfile.open(finalinput, std::ios_base::in);
// 				for(int i = 0; i < num_term; ++i){
// 					inputfile >> buffer;
// 					outputfile << buffer << " ";
// 				}
// 				outputfile << std::endl;
// 				inputfile.close();
// 			}
// 			outputfile.close();
// 		}
// 	}

// 	if (task == 3){ //Solver test
// 		stokesSolver refSolver(7, 1, 1);
// 		refSolver.forwardEqn();

// 		std::cout << "ref solution generated" << std::endl;

// 		double checkpoint[1][2];
// 		double checku[1][2];

// 		checkpoint[0][0] = 3.0/16.0;
// 		checkpoint[0][1] = 5.0/16.0;

// 		refSolver.getValues(checkpoint, checku, 1);
// 		std::cout << "ref: " << checku[0][0] << " " << checku[0][1] << std::endl;

// 		double locations[refSolver.num_node_Q1isoQ2][2];
// 		double u_compare[refSolver.num_node_Q1isoQ2][2];
// 		for (int i = 0; i < refSolver.num_node_Q1isoQ2; ++i){
// 			locations[i][0] = refSolver.points[0][i];
// 			locations[i][1] = refSolver.points[1][i];
// 		}

// 		std::cout << "set locations" << std::endl;

// 		Vec vdiff;
// 		VecCreate(MPI_COMM_SELF, &vdiff);
// 		VecSetSizes(vdiff, PETSC_DECIDE, 2*refSolver.num_node_Q1isoQ2);
// 		VecSetFromOptions(vdiff);

// 		Mat H1Matrix;
// 		IS isrow;
// 		IS iscol;
// 		int final_size = 2*refSolver.num_node_Q1isoQ2;
// 		PetscInt *indices;
// 		PetscMalloc1(final_size, &indices);
// 		for (int i = 0; i < final_size; i++){
// 			indices[i] = i;
// 		}
// 		ISCreateGeneral(PETSC_COMM_SELF, final_size, indices, PETSC_COPY_VALUES, &isrow);
// 		ISCreateGeneral(PETSC_COMM_SELF, final_size, indices, PETSC_COPY_VALUES, &iscol);
// 		MatCreateSubMatrix(refSolver.finalMatrix, isrow, iscol, MAT_INITIAL_MATRIX, &H1Matrix);


// 		stokesSolver L0Solver(0, 1, 1);
// 		L0Solver.forwardEqn();
// 		std::string L0solution = "L0usolution";
// 		writeU(L0Solver.u1.get(), L0Solver.u2.get(), L0Solver.num_node_Q1isoQ2, L0solution);


// 		L0Solver.getValues(checkpoint, checku, 1);
// 		L0Solver.getValues(locations, u_compare, refSolver.num_node_Q1isoQ2);

// 		for (int i = 0; i < refSolver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(vdiff, i, u_compare[i][0]-refSolver.u1[i], INSERT_VALUES);
// 			VecSetValue(vdiff, refSolver.num_node_Q1isoQ2+i, u_compare[i][1]-refSolver.u2[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(vdiff);
// 		VecAssemblyEnd(vdiff);

// 		Vec workspace;
// 	 	VecCreate(MPI_COMM_SELF, &workspace);
// 		VecSetSizes(workspace, PETSC_DECIDE, 2*refSolver.num_node_Q1isoQ2);
// 		VecSetFromOptions(workspace);
// 		VecSet(workspace, 0.0);
// 		VecAssemblyBegin(workspace);
// 		VecAssemblyEnd(workspace);
// 		double norm;
// 		MatMult(H1Matrix, vdiff, workspace);
// 		VecDot(vdiff, workspace, &norm);
// 		std::cout << "L0 norm: " << norm << std::endl;
// 		std::cout << "L0: " << checku[0][0] << " " << checku[0][1] << std::endl;

// 		stokesSolver L1Solver(1, 1, 1);
// 		L1Solver.forwardEqn();
// 		std::string L1solution = "L1usolution";
// 		writeU(L1Solver.u1.get(), L1Solver.u2.get(), L1Solver.num_node_Q1isoQ2, L1solution);


// 		L1Solver.getValues(checkpoint, checku, 1);
// 		L1Solver.getValues(locations, u_compare, refSolver.num_node_Q1isoQ2);

// 		for (int i = 0; i < refSolver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(vdiff, i, u_compare[i][0]-refSolver.u1[i], INSERT_VALUES);
// 			VecSetValue(vdiff, refSolver.num_node_Q1isoQ2+i, u_compare[i][1]-refSolver.u2[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(vdiff);
// 		VecAssemblyEnd(vdiff);
// 		MatMult(H1Matrix, vdiff, workspace);
// 		VecDot(vdiff, workspace, &norm);
// 		std::cout << "L1 norm: " << norm << std::endl;
//         std::cout << "L1: " << checku[0][0] << " " << checku[0][1] << std::endl;

// 		stokesSolver L2Solver(2, 1, 1);
// 		std::string L2solution = "L2usolution";
// 		L2Solver.forwardEqn();
// 		writeU(L2Solver.u1.get(), L2Solver.u2.get(), L2Solver.num_node_Q1isoQ2, L2solution);
// 		L2Solver.getValues(locations, u_compare, refSolver.num_node_Q1isoQ2);
//         L2Solver.getValues(checkpoint, checku, 1);

// 		for (int i = 0; i < refSolver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(vdiff, i, u_compare[i][0]-refSolver.u1[i], INSERT_VALUES);
// 			VecSetValue(vdiff, refSolver.num_node_Q1isoQ2+i, u_compare[i][1]-refSolver.u2[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(vdiff);
// 		VecAssemblyEnd(vdiff);
// 		MatMult(H1Matrix, vdiff, workspace);
// 		VecDot(vdiff, workspace, &norm);
// 		std::cout << "L2 norm: " << norm << std::endl;
//         std::cout << "L2: " << checku[0][0] << " " << checku[0][1] << std::endl;

// 		stokesSolver L3Solver(3, 1, 1);
// 		L3Solver.forwardEqn();
// 		std::string L3solution = "L3usolution";
// 		writeU(L3Solver.u1.get(), L3Solver.u2.get(), L3Solver.num_node_Q1isoQ2, L3solution);
// 		L3Solver.getValues(locations, u_compare, refSolver.num_node_Q1isoQ2);
//         L3Solver.getValues(checkpoint, checku, 1);
// 		for (int i = 0; i < refSolver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(vdiff, i, u_compare[i][0]-refSolver.u1[i], INSERT_VALUES);
// 			VecSetValue(vdiff, refSolver.num_node_Q1isoQ2+i, u_compare[i][1]-refSolver.u2[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(vdiff);
// 		VecAssemblyEnd(vdiff);
// 		MatMult(H1Matrix, vdiff, workspace);
// 		VecDot(vdiff, workspace, &norm);
// 		std::cout << "L3 norm: " << norm << std::endl;
//         std::cout << "L3: " << checku[0][0] << " " << checku[0][1] << std::endl;

// 		stokesSolver L4Solver(4, 1, 1);
// 		L4Solver.forwardEqn();
// 		std::string L4solution = "L4usolution";
// 		writeU(L4Solver.u1.get(), L4Solver.u2.get(), L4Solver.num_node_Q1isoQ2, L4solution);
// 		L4Solver.getValues(locations, u_compare, refSolver.num_node_Q1isoQ2);
//         L4Solver.getValues(checkpoint, checku, 1);
// 		for (int i = 0; i < refSolver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(vdiff, i, u_compare[i][0]-refSolver.u1[i], INSERT_VALUES);
// 			VecSetValue(vdiff, refSolver.num_node_Q1isoQ2+i, u_compare[i][1]-refSolver.u2[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(vdiff);
// 		VecAssemblyEnd(vdiff);
// 		MatMult(H1Matrix, vdiff, workspace);
// 		VecDot(vdiff, workspace, &norm);
// 		std::cout << "L4 norm: " << norm << std::endl;
//         std::cout << "L4: " << checku[0][0] << " " << checku[0][1] << std::endl;

// 		stokesSolver L5Solver(5, 1, 1);
// 		L5Solver.forwardEqn();
// 		std::string L5solution = "L5usolution";
// 		writeU(L5Solver.u1.get(), L5Solver.u2.get(), L5Solver.num_node_Q1isoQ2, L5solution);
// 		L5Solver.getValues(locations, u_compare, refSolver.num_node_Q1isoQ2);
//         L5Solver.getValues(checkpoint, checku, 1);
// 		for (int i = 0; i < refSolver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(vdiff, i, u_compare[i][0]-refSolver.u1[i], INSERT_VALUES);
// 			VecSetValue(vdiff, refSolver.num_node_Q1isoQ2+i, u_compare[i][1]-refSolver.u2[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(vdiff);
// 		VecAssemblyEnd(vdiff);
// 		MatMult(H1Matrix, vdiff, workspace);
// 		VecDot(vdiff, workspace, &norm);
// 		std::cout << "L5 norm: " << norm << std::endl;
//         std::cout << "L5: " << checku[0][0] << " " << checku[0][1] << std::endl;

// 		stokesSolver L6Solver(6, 1, 1);
// 		L6Solver.forwardEqn();
// 		L6Solver.getValues(locations, u_compare, refSolver.num_node_Q1isoQ2);
//         L6Solver.getValues(checkpoint, checku, 1);
// 		for (int i = 0; i < refSolver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(vdiff, i, u_compare[i][0]-refSolver.u1[i], INSERT_VALUES);
// 			VecSetValue(vdiff, refSolver.num_node_Q1isoQ2+i, u_compare[i][1]-refSolver.u2[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(vdiff);
// 		VecAssemblyEnd(vdiff);
// 		MatMult(H1Matrix, vdiff, workspace);
// 		VecDot(vdiff, workspace, &norm);
// 		std::cout << "L6 norm: " << norm << std::endl;
// 	}

// 	if (task == 4){ //Solver test
// 		double output;
// 		stokesSolver L0Solver(0, 1, 1);
// 		L0Solver.u[0] = 1;
// 		L0Solver.forwardEqn();
// 		L0Solver.int_vector();
// 		for (int i = 0; i < L0Solver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(L0Solver.states, L0Solver.num_node_Q1isoQ2+i, L0Solver.u1[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(L0Solver.states);
// 		VecAssemblyEnd(L0Solver.states);
// 		VecDot(L0Solver.intVec, L0Solver.states, &output);
// 		std::cout << "L0: " << output << std::endl;

// 		stokesSolver L1Solver(1, 1, 1);
// 		L1Solver.u[0] = 1;
// 		L1Solver.forwardEqn();
// 		L1Solver.int_vector();
//                 for (int i = 0; i < L1Solver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(L1Solver.states, L1Solver.num_node_Q1isoQ2+i, L1Solver.u1[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(L1Solver.states);
// 		VecAssemblyEnd(L1Solver.states);
// 		VecDot(L1Solver.intVec, L1Solver.states, &output);
// 		std::cout << "L1: " << output << std::endl;

// 		stokesSolver L2Solver(2, 1, 1);
// 		L2Solver.u[0] = 1;
// 		L2Solver.forwardEqn();
// 		L2Solver.int_vector();
//                 for (int i = 0; i < L2Solver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(L2Solver.states, L2Solver.num_node_Q1isoQ2+i, L2Solver.u1[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(L2Solver.states);
// 		VecAssemblyEnd(L2Solver.states);
// 		VecDot(L2Solver.intVec, L2Solver.states, &output);
// 		std::cout << "L2: " << output << std::endl;

// 		stokesSolver L3Solver(3, 1, 1);
// 		L3Solver.u[0] = 1;
// 		L3Solver.forwardEqn();
// 		L3Solver.int_vector();
//                 for (int i = 0; i < L3Solver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(L3Solver.states, L3Solver.num_node_Q1isoQ2+i, L3Solver.u1[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(L3Solver.states);
// 		VecAssemblyEnd(L3Solver.states);
// 		VecDot(L3Solver.intVec, L3Solver.states, &output);
// 		std::cout << "L3: " << output << std::endl;

// 		stokesSolver L4Solver(4, 1, 1);
// 		L4Solver.u[0] = 1;
// 		L4Solver.forwardEqn();
// 		L4Solver.int_vector();
//                 for (int i = 0; i < L4Solver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(L4Solver.states, L4Solver.num_node_Q1isoQ2+i, L4Solver.u1[i], INSERT_VALUES);	
// 		}
// 		VecAssemblyBegin(L4Solver.states);	
// 		VecAssemblyEnd(L4Solver.states);
// 		VecDot(L4Solver.intVec, L4Solver.states, &output);
// 		std::cout << "L4: " << output << std::endl;

// 		stokesSolver L5Solver(5, 1, 1);
// 		L5Solver.u[0] = 1;
// 		L5Solver.forwardEqn();
// 		L5Solver.int_vector();
//                 for (int i = 0; i < L5Solver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(L5Solver.states, L5Solver.num_node_Q1isoQ2+i, L5Solver.u1[i], INSERT_VALUES);	
// 		}
// 		VecAssemblyBegin(L5Solver.states);
// 		VecAssemblyEnd(L5Solver.states);
// 		VecDot(L5Solver.intVec, L5Solver.states, &output);
// 		std::cout << "L5: " << output << std::endl;

// 		stokesSolver L6Solver(6, 1, 1);
// 		L6Solver.u[0] = 1;
// 		L6Solver.forwardEqn();
// 		L6Solver.int_vector();
//                 for (int i = 0; i < L6Solver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(L6Solver.states, L6Solver.num_node_Q1isoQ2+i, L6Solver.u1[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(L6Solver.states);
// 		VecAssemblyEnd(L6Solver.states);
// 		VecDot(L6Solver.intVec, L6Solver.states, &output);
// 		std::cout << "L6: " << output << std::endl;

// 		stokesSolver L7Solver(7, 1, 1);
// 		L7Solver.u[0] = 1;
// 		L7Solver.forwardEqn();
// 		L7Solver.int_vector();
//                 for (int i = 0; i < L7Solver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(L7Solver.states, L7Solver.num_node_Q1isoQ2+i, L7Solver.u1[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(L7Solver.states);
// 		VecAssemblyEnd(L7Solver.states);
// 		VecDot(L7Solver.intVec, L7Solver.states, &output);
// 		std::cout << "L7: " << output << std::endl;

// 		stokesSolver L8Solver(8, 1, 1);
// 		L8Solver.u[0] = 1; 
// 		L8Solver.forwardEqn();
// 		L8Solver.int_vector();
//                 for (int i = 0; i < L8Solver.num_node_Q1isoQ2; ++i){
// 			VecSetValue(L8Solver.states, L8Solver.num_node_Q1isoQ2+i, L8Solver.u1[i], INSERT_VALUES);
// 		}
// 		VecAssemblyBegin(L8Solver.states);
// 		VecAssemblyEnd(L8Solver.states);
// 		VecDot(L8Solver.intVec, L8Solver.states, &output);
// 		std::cout << "L8: " << output << std::endl;
// 	}

// 	if (task == 5){ //Generate Observation
// 		obs_gen(8, rand_coef, obs, obsNumPerRow, noiseVariance);

// 		double pointwiseError;
// 		stokesSolver L0Solver(0, 1, 1);
// 		L0Solver.u[0] = 1;
// 		pointwiseError = L0Solver.solve(obs, obsNumPerRow*obsNumPerRow);
// 		std::cout << "L0: " << pointwiseError << std::endl;

// 		stokesSolver L1Solver(1, 1, 1);
// 		L1Solver.u[0] = 1;
// 		pointwiseError = L1Solver.solve(obs, obsNumPerRow*obsNumPerRow);
// 		std::cout << "L1: " << pointwiseError << std::endl;

// 		stokesSolver L2Solver(2, 1, 1);
// 		L2Solver.u[0] = 1;
// 		pointwiseError = L2Solver.solve(obs, obsNumPerRow*obsNumPerRow);
// 		std::cout << "L2: " << pointwiseError << std::endl;

// 		stokesSolver L3Solver(3, 1, 1);
// 		L3Solver.u[0] = 1;
// 		pointwiseError = L3Solver.solve(obs, obsNumPerRow*obsNumPerRow);
// 		std::cout << "L3: " << pointwiseError << std::endl;

// 		stokesSolver L4Solver(4, 1, 1);
// 		L4Solver.u[0] = 1;
// 		pointwiseError = L4Solver.solve(obs, obsNumPerRow*obsNumPerRow);
// 		std::cout << "L4: " << pointwiseError << std::endl;

// 		stokesSolver L5Solver(5, 1, 1);
// 		L5Solver.u[0] = 1;
// 		pointwiseError = L5Solver.solve(obs, obsNumPerRow*obsNumPerRow);
// 		std::cout << "L5: " << pointwiseError << std::endl;		

// 		stokesSolver L6Solver(6, 1, 1);
// 		L6Solver.u[0] = 1;
// 		pointwiseError = L6Solver.solve(obs, obsNumPerRow*obsNumPerRow);
// 		std::cout << "L6: " << pointwiseError << std::endl;	

//                 stokesSolver L7Solver(7, 1, 1);
// 		L7Solver.u[0] =  1;
//                 pointwiseError = L7Solver.solve(obs, obsNumPerRow*obsNumPerRow);
//                 std::cout << "L7: " << pointwiseError << std::endl;

// 	}


// 	if (task == 6){ //Generate Observation
// 		// double quadx[16] = {-0.9894, -0.9446, -0.8656, -0.7554, -0.6179, -0.4580, -0.2816, -0.0950,   0.0950,    0.2816,    0.4580,    0.6179,   0.7554,  0.8656,    0.9446,    0.9894};
// 		// double quadw[16] = {0.0272, 0.0623, 0.0952, 0.1246, 0.1496, 0.1692, 0.1826, 0.1895, 0.1895,    0.1826,    0.1692,    0.1496,    0.1246, 0.0952,    0.0623,    0.0272};
		
// 		stokesSolver refSolver(0, 1, 1);
// 		// double output;
			
// 		refSolver.u[0] = 1;
// 		refSolver.forwardEqn();
// 		// refSolver.int_vector();
// 		// for (int j = 0; j < refSolver.num_node_Q1isoQ2; ++j){
// 		// 	VecSetValue(refSolver.states, refSolver.num_node_Q1isoQ2+j, refSolver.u1[j], INSERT_VALUES);
// 		// }
// 		// VecAssemblyBegin(refSolver.states);
// 		// VecAssemblyEnd(refSolver.states);
// 		// VecDot(refSolver.intVec, refSolver.states, &output);
// 		// std::cout << rank << " " << output << std::endl;
// 	}
// 	PetscFinalize();
// 	return 0;
// }
