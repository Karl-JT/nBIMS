#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <chrono>
#include <memory>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <petsc.h>

#include <MLMCMC_Bi.h>
// #include <MLMCMC_Bi_Uniform.h>
#include <numericalRecipes.h>
#include <confIO.h>

double ref(double x, double y, int dir, void* ctx){
    double output = 0;
    if (dir == 0) {
		output = -1.0*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y)*(exp(1.0)-1.0);
    } else if (dir == 1) {
    	output = 1.0*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y)*(exp(1.0)-1.0);//*exp(1.0);
    }
    return output;
}

// double ref(double x, double y, int dir, void* ctx){
//     double output;
//     if (dir == 0) {
//     	output = 0.8*std::sin(2.*M_PI*y)*(exp(1.0)-1.0);//*exp(1.0);
//     } else if (dir == 1) {
// 		output = 0.8*std::sin(2.*M_PI*x)*(exp(1.0)-1.0);
//     }
//     return output;
// }

PetscErrorCode MyKSPMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *ctx)
{
	PetscPrintf(PETSC_COMM_SELF, "iteration %D KSP Residual norm %14.12e \n", n, rnorm);
	return 0;
}

#include "Q1isoQ2Mesh.h"

class stokesSolver {
private:
	std::unique_ptr<std::unique_ptr<double[]>[]> points;
	std::unique_ptr<std::unique_ptr<int[]>[]> quadrilaterals;
	std::unique_ptr<int[]> Q1_idx;
	std::unique_ptr<int[]> mesh_idx;
	std::unique_ptr<int[]> PBC_Q1isoQ2_idx;
	std::unique_ptr<int[]> PBC_Q1_idx;
	std::unique_ptr<int[]> bs;

	Mat massMatrix;
	Mat M;
	Mat A;
	Mat C;
	Mat B1;
	Mat B2;	
	Mat B1Cond;
	Mat B2Cond;
	Mat B1CondT;
	Mat B2CondT;
	Mat D;

	Mat zeroMatrix;
	Mat forwardOperatorFixed;
	Mat forwardOperator;
	Mat forwardOperatorCUDA;
	Mat Mfull;	
	Mat Cfull;

	IS pIdx;

	KSP    ksp;
	PC     pc;

public:
	Vec r;
	Vec states;
	Vec statesCUDA;
	Vec statesHalf;
	Vec statesP;
	Vec statesPfull;
	Vec V;
	Vec Vstab;
	Vec rhsF;
	Vec rhsCUDA;
	Vec intVecObs;
	Vec intVecQoi;

	int division0 = 2;
	int division_Q1isoQ2;
	int division_Q1;
	int num_Q1isoQ2_element;
	int num_Q1_element;
	int num_node_Q1isoQ2;
	int num_node_Q1;

	int stabOption = 0;
	int level;
	int timeSteps;
	int num_term;
	double nu = 0.1;
	double time = 0.0;
	double tMax = 1.0;
	double deltaT;
	// double obs=-1.513493402;
	double obs=-0.965649652758160;
	double noiseVariance;
	double beta = 1.0;
	PetscScalar *states_array;
	std::unique_ptr<double[]> samples;
	std::unique_ptr<double[]> u1;
	std::unique_ptr<double[]> u2;
	std::unique_ptr<double[]> uNorm;
	std::unique_ptr<double[]> pressure;
	std::unique_ptr<double[]> tauSUPG;
	std::unique_ptr<double[]> tauLSIC;

	int dummy=0;
	int counter=0;
	std::vector<double> readin;
	std::vector<double> phiList;

	std::default_random_engine generator;
	std::normal_distribution<double> normalDistribution{0.0, 1.0};
    std::uniform_real_distribution<double> uniformDistribution{-1.0, 1.0};

	stokesSolver(int level_, int num_term_, double noiseVariance_);
	~stokesSolver(){
		MatDestroy(&massMatrix);
		MatDestroy(&M);
		MatDestroy(&A);
		MatDestroy(&B1);
		MatDestroy(&B2);
		MatDestroy(&B1Cond);
		MatDestroy(&B2Cond);
		MatDestroy(&C);
		MatDestroy(&D);
		MatDestroy(&B1CondT);
		MatDestroy(&B2CondT);

		MatDestroy(&zeroMatrix);
		MatDestroy(&forwardOperator);
		MatDestroy(&forwardOperatorFixed);
		MatDestroy(&forwardOperatorCUDA);
		MatDestroy(&Mfull);	
		MatDestroy(&Cfull);

		VecDestroy(&r);
		VecDestroy(&V);
		VecDestroy(&Vstab);
		VecDestroy(&states);	
		VecDestroy(&statesP);	
		VecDestroy(&statesPfull);	
		VecDestroy(&statesCUDA);
		VecDestroy(&statesHalf);
		VecDestroy(&rhsCUDA);
		VecDestroy(&rhsF);
		VecDestroy(&intVecObs);
		VecDestroy(&intVecQoi);

		ISDestroy(&pIdx);
		KSPDestroy(&ksp);
	};

	void mat_init(Mat &xMat, int m, int n);
	void solver_init();

	void shape_function(double epsilon, double eta, double N[]);
	void jacobian_matrix(double x1[4], double x2[4], double epsilon, double eta, double J[2][2]);
	void jacobian_inv(double x1[4], double x2[4], double epsilon, double eta, double J[2][2]);
	void basis_function(double epsilon, double eta, double N[]);
	void basis_function_derivative(double dPhi[4][2], double epsilon, double eta);
	void hat_function_derivative(double dPhi[4][2], double epsilon, double eta, double x1[4], double x2[4]);

	double fsource(double x, double y, int dir);
	double shape_interpolation(double epsilon, double eta, double x[4]);
	double jacobian_det(double J[2][2]);

	void mass_matrix_element(double P[][4], double A[][4]);
	void stiffness_matrix_element(double P[][4], double A[][4]);
	void b1_matrix_element(double x1[4], double x2[4], double b1[4][4]);
	void b2_matrix_element(double x1[4], double x2[4], double b2[4][4]);
	void tauA11_matrix_element(double x1[4], double x2[4], double tauA11[4][4]);
	void tauA12_matrix_element(double x1[4], double x2[4], double tauA12[4][4]);
	void tauA21_matrix_element(double x1[4], double x2[4], double tauA21[4][4]);
	void tauA22_matrix_element(double x1[4], double x2[4], double tauA22[4][4]);
	void c_matrix_element(double x1[4], double x2[4], double ux[4], double uy[4], double c[4][4]);
	void S_matrix_element(double x1[4], double x2[4], double ux[4], double uy[4], double s_element[4][4], int dir);

	void M_matrix();
	void A_matrix();
	void tauA_matrix();
	void B1Cond_matrix();
	void B2Cond_matrix();
	void B1TCond_matrix();
	void B2TCond_matrix();
	void C_matrix();
	void D_matrix();
	void G_matrix();

	void tauUpdate();
	void load_vector_element(double x1[], double x2[], double vx[], double vy[]);
	void load_vector();
	void load_vector_element_stab(double x1[], double x2[], double ux[], double uy[], double v[], int dir);
	void load_vector_stab();
	void linear_system_setup();
	void system_update_function();
	void system_update_jacobian();

	void FormFunction(SNES snes, Vec x, Vec f);
	void FormJacobian(SNES snes, Vec x, Mat jac, Mat B);
	void forwardStep();
	void solver_setup();
	void updateGeneratorSeed(double seed_);
	void int_vector_element(double x1[], double x2[], double vx[], double vy[], double expCoef);
	double lnLikelihood();
	double solve4Obs();
	double solve4QoI();
	double obsOutput();
	double qoiOutput();
	void int_vector_obs(double expCoef);
	void int_vector_qoi(double expCoef);
	double getValues(double x, double y, int dir);
	void solve(int flag=0);
	void interpolation_element(double x1[], double x2[], double v[], int dir);
	void interpolation_vector(Vec interpolation);
	void interpolate(Vec solutionInterpolation);
	void priorSample(double initialSamples[]);
	void H1test();

	void readData(int mainL, int auxL, int Mnum, int color);
	void turnondummy(int mainL_, int auxL_, int Mnum_, int color_);

	static PetscErrorCode FormFunctionStatic(SNES snes, Vec x, Vec f, void *ctx){
		stokesSolver *ptr = static_cast<stokesSolver*>(ctx);
		ptr->FormFunction(snes, x, f);
		return 0;
	}

	static PetscErrorCode FormJacobianStatic(SNES snes, Vec x, Mat jac, Mat B, void *ctx){
		stokesSolver *ptr = static_cast<stokesSolver*>(ctx);
		ptr->FormJacobian(snes, x, jac, B);
		return 0;
	}

	static double getValues_static(double x, double y, int dir, void *ctx){
		double output;
		stokesSolver *ptr = static_cast<stokesSolver*>(ctx);
		output = ptr->getValues(x, y, dir);
		return output;
	}
};

stokesSolver::stokesSolver(int level_, int num_term_, double noiseVariance_) : level(level_), num_term(num_term_), noiseVariance(noiseVariance_) {
	samples = std::make_unique<double[]>(num_term_);
	timeSteps = std::pow(2, level_+1);
	deltaT = tMax/timeSteps;

	division_Q1isoQ2 = division0*(std::pow(2, level_+1));
	division_Q1 = division_Q1isoQ2/2;
	num_Q1isoQ2_element = division_Q1isoQ2*division_Q1isoQ2;
	num_Q1_element = 0.25*num_Q1isoQ2_element;
	num_node_Q1isoQ2 = std::pow(division_Q1isoQ2+1, 2);
	num_node_Q1 = std::pow(division_Q1isoQ2/2+1, 2);

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

	solver_init();

	uNorm = std::make_unique<double[]>(num_node_Q1isoQ2);
	u1    = std::make_unique<double[]>(num_node_Q1isoQ2);
	u2    = std::make_unique<double[]>(num_node_Q1isoQ2);
	VecGetArray(states, &states_array);
	std::copy(states_array, states_array+num_node_Q1isoQ2, u1.get());
	std::copy(states_array+num_node_Q1isoQ2, states_array+2*num_node_Q1isoQ2, u2.get());
	VecRestoreArray(states, &states_array);
	for (int i = 0; i < num_node_Q1isoQ2; ++i){
		uNorm[i] = sqrt(pow(u1[i], 2)+pow(u1[i], 2));
	}
	tauSUPG = std::make_unique<double[]>(num_node_Q1isoQ2);
	tauLSIC = std::make_unique<double[]>(num_node_Q1isoQ2);
	tauUpdate();

	linear_system_setup();
	int idxP[num_node_Q1];
	for (int i = 0; i < num_node_Q1; ++i){
		idxP[i] = 2*num_node_Q1isoQ2+i;
	}
	ISCreateGeneral(PETSC_COMM_SELF, num_node_Q1, idxP, PETSC_COPY_VALUES, &pIdx);
	VecGetSubVector(states, pIdx, &statesP);
	MatMult(D, statesP, statesPfull);

	pressure = std::make_unique<double[]>(num_node_Q1isoQ2);
	VecGetArray(statesPfull, &states_array);
	std::copy(states_array, states_array+num_node_Q1isoQ2, pressure.get());
	VecRestoreArray(statesPfull, &states_array);

	solver_setup();
};

void stokesSolver::mat_init(Mat &xMat, int m, int n){
	MatCreate(PETSC_COMM_SELF, &xMat);
	MatSetSizes(xMat, PETSC_DECIDE, PETSC_DECIDE, m, n);
	MatSetType(xMat, MATSEQAIJ);
	MatSeqAIJSetPreallocation(xMat, 12, NULL);
	MatMPIAIJSetPreallocation(xMat, 12, NULL, 12, NULL);
	int count = std::min(m, n);
	for (int i = 0; i < count; ++i){
		MatSetValue(xMat, i, i, 0, INSERT_VALUES);
	}
	MatAssemblyBegin(xMat, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(xMat, MAT_FINAL_ASSEMBLY);
}

void stokesSolver::solver_init(){
 	VecCreate(PETSC_COMM_SELF, &states);
	VecSetSizes(states, PETSC_DECIDE, 2*num_node_Q1isoQ2+num_node_Q1);
	VecSetType(states, VECSEQ);
	VecSet(states,0.0);
	VecAssemblyBegin(states);
	VecAssemblyEnd(states);

	VecDuplicate(states, &statesHalf);
	VecDuplicate(states, &r);
	VecDuplicate(states, &V);
	VecDuplicate(states, &Vstab);
	VecDuplicate(states, &rhsF);
	VecDuplicate(states, &intVecObs);
	VecDuplicate(states, &intVecQoi);

 	VecCreate(PETSC_COMM_SELF, &statesCUDA);
	VecSetSizes(statesCUDA, PETSC_DECIDE, 2*num_node_Q1isoQ2+num_node_Q1);
	if (level > 10){
		VecSetType(statesCUDA, VECCUDA);
	} else {
		VecSetType(statesCUDA, VECSEQ);
	}
	VecSet(statesCUDA,0.0);
	VecAssemblyBegin(statesCUDA);
	VecAssemblyEnd(statesCUDA);

	VecDuplicate(statesCUDA, &rhsCUDA);

 	VecCreate(PETSC_COMM_SELF, &statesPfull);
	VecSetSizes(statesPfull, PETSC_DECIDE, num_node_Q1isoQ2);
	VecSetType(statesPfull, VECSEQ);
	VecSet(statesPfull, 0.0);
	VecAssemblyBegin(statesPfull);
	VecAssemblyEnd(statesPfull);

	mat_init(massMatrix, num_node_Q1isoQ2, num_node_Q1isoQ2);
	mat_init(A, num_node_Q1isoQ2, num_node_Q1isoQ2);
	mat_init(B1, num_node_Q1isoQ2, num_node_Q1isoQ2);
	mat_init(B2, num_node_Q1isoQ2, num_node_Q1isoQ2);

	mat_init(D, num_node_Q1isoQ2, num_node_Q1);
	mat_init(C, num_node_Q1isoQ2, num_node_Q1isoQ2);

	mat_init(Cfull, num_node_Q1, num_node_Q1);
	mat_init(zeroMatrix, num_node_Q1, num_node_Q1);
	// mat_init(forwardOperator, 2*num_node_Q1isoQ2+num_node_Q1, 2*num_node_Q1isoQ2+num_node_Q1);

	MatCreate(PETSC_COMM_SELF, &forwardOperatorCUDA);
	MatSetSizes(forwardOperatorCUDA, PETSC_DECIDE, PETSC_DECIDE, 2*num_node_Q1isoQ2+num_node_Q1, 2*num_node_Q1isoQ2+num_node_Q1);
	if (level > 10){
		MatSetType(forwardOperatorCUDA, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(forwardOperatorCUDA, MATSEQAIJ);
	}
	MatSeqAIJSetPreallocation(forwardOperatorCUDA, 12, NULL);
	MatMPIAIJSetPreallocation(forwardOperatorCUDA, 12, NULL, 12, NULL);
	for (int i = 0; i < 2*num_node_Q1isoQ2+num_node_Q1; ++i){
		MatSetValue(forwardOperatorCUDA, i, i, 0, INSERT_VALUES);
	}
	MatAssemblyBegin(forwardOperatorCUDA, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(forwardOperatorCUDA, MAT_FINAL_ASSEMBLY);
}

void stokesSolver::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};

// double stokesSolver::fsource(double x, double y, int dir){
//     double output = 0;
//     if (dir == 0) {
// 		output = -samples[0]*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y)*exp(time) - 2.*samples[0]*pow(2.*M_PI, 2)*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y)*(exp(time)-1.0) - samples[0]*samples[0]*M_PI*std::sin(4.0*M_PI*x)*pow(exp(time)-1, 2.); //samples[0]*
//     } else if (dir == 1) {
//     	output = samples[0]*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y)*exp(time) + 2.*samples[0]*pow(2.*M_PI, 2)*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y)*(exp(time)-1.0) - samples[0]*samples[0]*M_PI*std::sin(4.0*M_PI*y)*pow(exp(time)-1, 2.); //samples[0]*
//     }
//     return output;
// }

double stokesSolver::fsource(double x, double y, int dir){
    double output;
    if (dir == 0) {
    	output = -samples[0]*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y)*exp(time);//-samples[1]*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y)*exp(time)-samples[2]/2.*std::cos(4.*M_PI*x)*std::sin(4.*M_PI*y)*exp(time)-samples[3]/2.*std::sin(4.*M_PI*x)*std::cos(4.*M_PI*y)*exp(time);
    } else if (dir == 1) {
    	output = samples[0]*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y)*exp(time);//+samples[1]*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y)*exp(time)+samples[2]/2.*std::sin(4.*M_PI*x)*std::cos(4.*M_PI*y)*exp(time)+samples[3]/2.*std::cos(4.*M_PI*x)*std::sin(4.*M_PI*y)*exp(time);
    } else {
    	std::cout << "forcing direction only available in 0, 1" << std::endl;
    	output = 0;
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

void stokesSolver::tauUpdate(){
	double Pe;
	for (int i = 0; i < num_node_Q1isoQ2; ++i){
		Pe = uNorm[i]*1.0/division_Q1isoQ2/2.0/nu;
		if (Pe < 1){
			tauSUPG[i] = pow(1.0/division_Q1isoQ2, 2)/12.0/nu;
		} else {
			tauSUPG[i] = 1.0/division_Q1isoQ2/2.0/uNorm[i]*(std::cosh(Pe)/std::sinh(Pe)-1.0/Pe);
		}
		tauLSIC[i] = uNorm[i]*1.0/division_Q1isoQ2/2.0;
	}
}

void stokesSolver::M_matrix(){
	PetscBool flag;
	MatAssembled(massMatrix, &flag);
	if (flag){
		MatResetPreallocation(massMatrix);
		MatZeroEntries(massMatrix);
	}
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
 				MatSetValue(massMatrix, PBC_Q1isoQ2_idx[quadrilaterals[m][i]], PBC_Q1isoQ2_idx[quadrilaterals[n][i]], mass_element[m][n], ADD_VALUES);
			}
		}			
	}
	MatAssemblyBegin(massMatrix, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(massMatrix, MAT_FINAL_ASSEMBLY);
	MatDuplicate(massMatrix, MAT_COPY_VALUES, &M);
	MatScale(M, 1.0/deltaT);

	Mat list[] = {M, NULL, NULL, NULL, M, NULL, NULL, NULL, zeroMatrix};
	MatCreateNest(PETSC_COMM_SELF, 3, NULL, 3, NULL, list, &Mfull);
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
	PetscBool flag;
	MatAssembled(A, &flag);
	if (flag){
		MatResetPreallocation(A);
		MatZeroEntries(A);
	}

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
 				MatSetValue(A, PBC_Q1isoQ2_idx[quadrilaterals[m][i]], PBC_Q1isoQ2_idx[quadrilaterals[n][i]], nu*stiffness_element[m][n], ADD_VALUES);
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

void stokesSolver::B1Cond_matrix(){
	PetscBool flag;
	MatAssembled(B1, &flag);
	if (flag){
		MatResetPreallocation(B1);
		MatZeroEntries(B1);
	} else {
        MatZeroEntries(B1);
    }

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
				MatSetValue(B1, PBC_Q1_idx[quadrilaterals[m][i]], PBC_Q1isoQ2_idx[quadrilaterals[n][i]], b1_element[m][n], ADD_VALUES);
            }
		}
	}
	MatAssemblyBegin(B1, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B1, MAT_FINAL_ASSEMBLY);	

	MatTransposeMatMult(D, B1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &B1Cond);
};

void stokesSolver::B2Cond_matrix(){
	PetscBool flag;
	MatAssembled(B2, &flag);
	if (flag){
		MatResetPreallocation(B2);
		MatZeroEntries(B2);
	}
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
				MatSetValue(B2, PBC_Q1_idx[quadrilaterals[m][i]], PBC_Q1isoQ2_idx[quadrilaterals[n][i]], b2_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(B2, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B2, MAT_FINAL_ASSEMBLY);		
	MatTransposeMatMult(D, B2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &B2Cond);
};

void stokesSolver::B1TCond_matrix(){
	MatTranspose(B1Cond, MAT_INITIAL_MATRIX, &B1CondT);
}

void stokesSolver::B2TCond_matrix(){
	MatTranspose(B2Cond, MAT_INITIAL_MATRIX, &B2CondT);	
}

void stokesSolver::c_matrix_element(double x1[4], double x2[4], double ux[4], double uy[4], double c[4][4]){
	double dPhi[4][2];
	double J[2][2];
	double N[4];

	double refPoints[2];
	double refWeights[2];

	refPoints[0] = -1./sqrt(3.0);
	refPoints[1] =  1./sqrt(3.0);

	refWeights[0] = 1.0;
	refWeights[1] = 1.0;

	double gpPoints0x[4];
	double gpPoints0y[4];
	double gpPoints1x[4][4];
	double gpPoints1y[4][4];
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
				gpPoints1x[2*i+j][k] = dPhi[k][0];
				gpPoints1y[2*i+j][k] = dPhi[k][1];
				gpPoints2[2*i+j][k] = N[k];
			}
			gpPoints0x[2*i+j] = shape_interpolation(refPoints[i], refPoints[j], ux);
			gpPoints0y[2*i+j] = shape_interpolation(refPoints[i], refPoints[j], uy);
        }
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			c[i][j] = 0;
			for (int k = 0; k < 4; k++){
				c[i][j] += gpWeights[k]*(gpPoints0x[k]*gpPoints1x[k][j]*gpPoints2[k][i]+gpPoints0y[k]*gpPoints1y[k][j]*gpPoints2[k][i])*Jdet[k];
			}
		}
	}
};

void stokesSolver::C_matrix(){
	PetscBool flag;
	MatAssembled(C, &flag);
	if (flag){
		MatResetPreallocation(C);
		MatZeroEntries(C);
	}
	double c_element[4][4];
	double element_points[2][4];
	double ux[4];
	double uy[4];
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
			ux[j] = u1[PBC_Q1isoQ2_idx[quadrilaterals[j][i]]];
			uy[j] = u2[PBC_Q1isoQ2_idx[quadrilaterals[j][i]]];
		}
		c_matrix_element(element_points[0], element_points[1], ux, uy, c_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
				MatSetValue(C, PBC_Q1isoQ2_idx[quadrilaterals[m][i]], PBC_Q1isoQ2_idx[quadrilaterals[n][i]], c_element[m][n], ADD_VALUES);
			}
		}
	}
	MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);	

	MatAssembled(Cfull, &flag);
	if (flag){
		MatDestroy(&Cfull);
	}

	Mat Cnest;
	Mat list[] = {C, NULL, NULL, NULL, C, NULL, NULL, NULL, zeroMatrix};
	MatCreateNest(PETSC_COMM_SELF, 3, NULL, 3, NULL, list, &Cnest);	
	MatConvert(Cnest, "seqaij", MAT_INITIAL_MATRIX, &Cfull);
	MatDestroy(&Cnest);
};

void stokesSolver::D_matrix(){
	PetscBool flag;
	MatAssembled(D, &flag);
	if (flag){
		MatResetPreallocation(D);
		MatZeroEntries(D);
	}

	// for (int i = 0; i < num_node_Q1; i++){
	// 	MatSetValue(D, i, i, 1, INSERT_VALUES);
	// }
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		if (Q1_idx[quadrilaterals[0][i]] < num_node_Q1isoQ2){
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[0][i]], PBC_Q1_idx[quadrilaterals[0][i]], 1.0, INSERT_VALUES);
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[1][i]], PBC_Q1_idx[quadrilaterals[0][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[3][i]], PBC_Q1_idx[quadrilaterals[0][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[2][i]], PBC_Q1_idx[quadrilaterals[0][i]], 0.25, INSERT_VALUES);
		} else if (Q1_idx[quadrilaterals[1][i]] < num_node_Q1isoQ2){
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[1][i]], PBC_Q1_idx[quadrilaterals[1][i]], 1.0, INSERT_VALUES);
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[2][i]], PBC_Q1_idx[quadrilaterals[1][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[0][i]], PBC_Q1_idx[quadrilaterals[1][i]], 0.5, INSERT_VALUES);			
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[3][i]], PBC_Q1_idx[quadrilaterals[1][i]], 0.25, INSERT_VALUES);
		} else if (Q1_idx[quadrilaterals[2][i]] < num_node_Q1isoQ2){
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[2][i]], PBC_Q1_idx[quadrilaterals[2][i]], 1.0, INSERT_VALUES);
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[3][i]], PBC_Q1_idx[quadrilaterals[2][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[1][i]], PBC_Q1_idx[quadrilaterals[2][i]], 0.5, INSERT_VALUES);			
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[0][i]], PBC_Q1_idx[quadrilaterals[2][i]], 0.25, INSERT_VALUES);
		} else if (Q1_idx[quadrilaterals[3][i]] < num_node_Q1isoQ2){
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[3][i]], PBC_Q1_idx[quadrilaterals[3][i]], 1.0, INSERT_VALUES);
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[0][i]], PBC_Q1_idx[quadrilaterals[3][i]], 0.5, INSERT_VALUES);
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[2][i]], PBC_Q1_idx[quadrilaterals[3][i]], 0.5, INSERT_VALUES);			
			MatSetValue(D, PBC_Q1_idx[quadrilaterals[1][i]], PBC_Q1_idx[quadrilaterals[3][i]], 0.25, INSERT_VALUES);
		}
	}
	MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);
};

void stokesSolver::load_vector_element(double x1[], double x2[], double vx[], double vy[]){
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
			gpPoints1x[5*i+j] = fsource(x, y, 0);
			gpPoints1y[5*i+j] = fsource(x, y, 1);
			for (int k = 0; k < 4; ++k){
				gpPoints2[5*i+j][k] = N[k];
			}
		}
	}
	for (int i = 0; i < 4; i++){
		vx[i] = 0;
		vy[i] = 0;
		for (int k = 0; k < 25; k++){
			vx[i] += gpWeights[k]*gpPoints1x[k]*gpPoints2[k][i]*Jdet[k];
			vy[i] += gpWeights[k]*gpPoints1y[k]*gpPoints2[k][i]*Jdet[k];
		}
	}
};

void stokesSolver::load_vector(){
	double v_elementx[4];
	double v_elementy[4];
	double element_points[2][4];
	VecZeroEntries(V);
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		load_vector_element(element_points[0], element_points[1], v_elementx, v_elementy);
		for (int k = 0; k < 4; k++){
			VecSetValue(V, PBC_Q1isoQ2_idx[quadrilaterals[k][i]], v_elementx[k], ADD_VALUES);
			VecSetValue(V, num_node_Q1isoQ2+PBC_Q1isoQ2_idx[quadrilaterals[k][i]], v_elementy[k], ADD_VALUES);
		}
	}
	VecAssemblyBegin(V);
	VecAssemblyEnd(V);
};

void stokesSolver::load_vector_element_stab(double x1[], double x2[], double ux[], double uy[], double v[], int dir){
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

	double gpPoints0x[25];
	double gpPoints0y[25];
	double gpPoints1[25];
	double gpPoints2x[25][4];
	double gpPoints2y[25][4];
	double gpWeights[25];
	double Jdet[25];

	double x;
	double y;

	for (int i = 0; i < 5; i++){
		for (int j = 0; j < 5; j++){
			basis_function(refPoints[i], refPoints[j], N);
			jacobian_matrix(x1, x2, refPoints[i], refPoints[j], J);
			hat_function_derivative(dPhi, refPoints[i], refPoints[j], x1, x2);
			x = shape_interpolation(refPoints[i], refPoints[j], x1);
			y = shape_interpolation(refPoints[i], refPoints[j], x2);
			gpWeights[5*i+j] = refWeights[i]*refWeights[j];
			Jdet[5*i+j] = jacobian_det(J);
			gpPoints1[5*i+j] = fsource(x, y, dir);
			for (int k = 0; k < 4; ++k){
				gpPoints2x[5*i+j][k] = dPhi[k][0];
				gpPoints2y[5*i+j][k] = dPhi[k][1];
			}
			gpPoints0x[5*i+j] = shape_interpolation(refPoints[i], refPoints[j], ux);
			gpPoints0y[5*i+j] = shape_interpolation(refPoints[i], refPoints[j], uy);
		}
	}
	for (int i = 0; i < 4; i++){
		v[i] = 0;
		for (int k = 0; k < 25; k++){
			v[i] += gpWeights[k]*gpPoints1[k]*(gpPoints0x[k]*gpPoints2x[k][i]+gpPoints0y[k]*gpPoints2y[k][i])*Jdet[k];
		}
	}
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
			vx[i] += gpWeights[k]*gpPoints1x[k]*gpPoints1y[k]*gpPoints2y[k][i]*Jdet[k];
			vy[i] += -gpWeights[k]*gpPoints1x[k]*gpPoints1y[k]*gpPoints2x[k][i]*Jdet[k];
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

void stokesSolver::linear_system_setup(){
	M_matrix();
	A_matrix();
	D_matrix();
	B1Cond_matrix();
	B2Cond_matrix();
	B1TCond_matrix();
	B2TCond_matrix();

	MatAXPY(A, 1.0, M, DIFFERENT_NONZERO_PATTERN);
	Mat listFunction[] = {A, NULL, B1CondT, NULL, A, B2CondT, B1Cond, B2Cond, NULL};
	MatCreateNest(PETSC_COMM_SELF, 3, NULL, 3, NULL, listFunction, &forwardOperatorFixed);
	MatConvert(forwardOperatorFixed, "seqaij", MAT_INITIAL_MATRIX, &forwardOperator);
	MatAXPY(forwardOperator, 1.0, forwardOperatorCUDA, DIFFERENT_NONZERO_PATTERN);
	MatCopy(forwardOperator, forwardOperatorCUDA, DIFFERENT_NONZERO_PATTERN);
}

//Iterative solver
void stokesSolver::solver_setup(){
	KSPCreate(PETSC_COMM_SELF, &ksp);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCFIELDSPLIT);
	PCFieldSplitSetDetectSaddlePoint(pc, PETSC_TRUE);
	PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, NULL);
	// KSPMonitorSet(ksp, MyKSPMonitor, NULL, 0);
	KSPSetTolerances(ksp, 1e-8, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetFromOptions(ksp);
	KSPSetPC(ksp, pc);
}

// //Direct Solver
// void stokesSolver::solver_setup(){
// 	KSPCreate(PETSC_COMM_SELF, &ksp);
// 	KSPSetFromOptions(ksp);
// }

void stokesSolver::forwardStep(){
	C_matrix();
	MatCopy(forwardOperator, forwardOperatorCUDA, SAME_NONZERO_PATTERN);
	MatAXPY(forwardOperatorCUDA, 1.0, Cfull, SUBSET_NONZERO_PATTERN);

	// const char fname[] = "A_matrix";
	// PetscViewer matrixSave;
	// PetscViewerCreate(PETSC_COMM_SELF, &matrixSave);
	// PetscViewerBinaryOpen(PETSC_COMM_SELF, fname, FILE_MODE_WRITE, &matrixSave);
	// MatView(forwardOperatorCUDA, matrixSave);
	// PetscViewerDestroy(&matrixSave);
	KSPSetOperators(ksp, forwardOperatorCUDA, forwardOperatorCUDA);

	load_vector();
	MatMultAdd(Mfull, states, V, rhsCUDA);
    // VecView(rhsCUDA,PETSC_VIEWER_STDOUT_WORLD);

	// const char vname[] = "b_vector";
	// PetscViewer vectorSave;
	// PetscViewerCreate(PETSC_COMM_SELF, &vectorSave);
	// PetscViewerBinaryOpen(PETSC_COMM_SELF, vname, FILE_MODE_WRITE, &vectorSave);
	// VecView(rhsCUDA, vectorSave);
	// PetscViewerDestroy(&vectorSave);
	KSPSolve(ksp, rhsCUDA, statesCUDA);
	VecCopy(statesCUDA, states);

	VecGetArray(states, &states_array);
	std::copy(states_array, states_array+num_node_Q1isoQ2, u1.get());
	std::copy(states_array+num_node_Q1isoQ2, states_array+2*num_node_Q1isoQ2, u2.get());
	VecRestoreArray(states, &states_array);
	for (int i = 0; i < num_node_Q1isoQ2; ++i){
		uNorm[i] = sqrt(pow(u1[i],2) + pow(u2[i],2));
	}
};

void stokesSolver::solve(int flag)
{
	if (dummy == 1){
		counter++;
		samples[0]=readin[counter];
		return;
	}
	// std::cout << "solving: m=" << samples[0] << std::endl;
	VecSet(states,0.0);
	time = 0.0;
	for (int i = 0; i < timeSteps; ++i){
		std::cout << "#################" << " level " << level << ", step " << i+1 << " #################" << std::endl;
		std::clock_t c_start = std::clock();
		auto wcts = std::chrono::system_clock::now();		

		time = time+deltaT;
		forwardStep();
		// VecView(states, PETSC_VIEWER_STDOUT_WORLD);

		if (abs(time - 0.5) <  1e-6){
			VecCopy(states, statesHalf);
			solve4QoI();
			if (flag == 1){
				return;
			}
		}		
		std::clock_t c_end = std::clock();
		double time_elapsed_ms = (c_end-c_start)/ (double)CLOCKS_PER_SEC;
		std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
		std::cout << "wall time " << wctduration.count() << " cpu  time: " << time_elapsed_ms << std::endl;
	}
	// H1test();
	// solve4QoI();
	solve4Obs();
	// double QOIres = qoiOutput();
	// double OBSres = obsOutput();
	// std::cout << samples[0] << " " << QOIres << " " << OBSres << std::endl;
}

void stokesSolver::priorSample(double initialSamples[]){
	initialSamples[0] = normalDistribution(generator);
	// initialSamples[0] = uniformDistribution(generator);
}

double stokesSolver::getValues(double x, double y, int dir)
{
    double element_h = 1.0/division_Q1isoQ2;

    int column;
    int row;

    int square[4];
    double coords[4];

    double epsilon;
    double eta;

    double local_u[4] = {0};

    column  = x / element_h;
    row     = y / element_h;

    if (column == division_Q1isoQ2){
            column -= 1;
    }
    if (row == division_Q1isoQ2){
            row -= 1;
    }

    square[0] = quadrilaterals[0][division_Q1isoQ2*row+column];
    square[1] = quadrilaterals[1][division_Q1isoQ2*row+column];
    square[2] = quadrilaterals[2][division_Q1isoQ2*row+column];
    square[3] = quadrilaterals[3][division_Q1isoQ2*row+column];

    coords[0] = points[0][square[0]];
    coords[1] = points[0][square[2]];
    coords[2] = points[1][square[0]];
    coords[3] = points[1][square[2]];

    epsilon = (x-coords[0])/(coords[1]-coords[0])*2-1;
    eta     = (y-coords[2])/(coords[3]-coords[2])*2-1;

    for (int k = 0; k < 4; ++k){
        if (dir == 0){
	        local_u[k] = u1[PBC_Q1isoQ2_idx[square[k]]];
        } else if (dir == 1){
	        local_u[k] = u2[PBC_Q1isoQ2_idx[square[k]]];        	
        }
    }
    double velocity = shape_interpolation(epsilon, eta, local_u);
	return velocity;  
}

double stokesSolver::obsOutput(){
		double obs;
		VecDot(intVecObs, states, &obs);
		obs = 100.*obs;
		return obs;	
}

double stokesSolver::qoiOutput(){
		double qoi;
		// VecView(statesHalf, PETSC_VIEWER_STDOUT_WORLD);
		// VecDot(intVecQoi, states, &qoi);
		VecDot(intVecQoi, statesHalf, &qoi);
		qoi = 100.*qoi;
		return qoi;
}

double stokesSolver::solve4Obs(){
	int_vector_obs(0.5);
	return obsOutput();
}

double stokesSolver::solve4QoI(){
	int_vector_qoi(1.5);
	return qoiOutput();
}

double stokesSolver::lnLikelihood(){
	if (dummy == 1){
		return phiList[counter-1];
	}
	double obsResult = obsOutput();
	double lnLikelihood = -0.5/noiseVariance*pow(obsResult-obs,2);
	return lnLikelihood;
}

void stokesSolver::readData(int mainL, int auxL, int Mnum, int color){
	if (mainL==4 && auxL==5){
		std::string pathName = "L54L4Samples.txt";
		std::ifstream myfile(pathName);
		std::string line;
		double temp;

		std::getline(myfile, line);
		std::istringstream iss(line);
		while (iss >> temp){
			readin.push_back(temp);
		}

		std::getline(myfile, line);
		std::istringstream iss1(line);
		while (iss1 >> temp){
			phiList.push_back(temp);
		}	

		counter = color*Mnum;
	}else if (mainL==5 && auxL==4){
		std::string pathName = "L54L5Samples.txt";
		std::ifstream myfile(pathName);
		std::string line;
		double temp;

		std::getline(myfile, line);
		std::istringstream iss(line);
		while (iss >> temp){
			readin.push_back(temp);
		}

		std::getline(myfile, line);
		std::istringstream iss1(line);
		while (iss1 >> temp){
			phiList.push_back(temp);
		}	

		counter = color*Mnum;
	}else if (mainL==6 && auxL==5){
		std::string pathName = "L65L6Samples.txt";
		std::ifstream myfile(pathName);
		std::string line;
		double temp;

		std::getline(myfile, line);
		std::istringstream iss(line);
		while (iss >> temp){
			readin.push_back(temp);
		}

		std::getline(myfile, line);
		std::istringstream iss1(line);
		while (iss1 >> temp){
			phiList.push_back(temp);
		}	

		counter = color*Mnum;
	}else if (mainL==5 && auxL==6){
		std::string pathName = "L65L5Samples.txt";
		std::ifstream myfile(pathName);
		std::string line;
		double temp;

		std::getline(myfile, line);
		std::istringstream iss(line);
		while (iss >> temp){
			readin.push_back(temp);
		}

		std::getline(myfile, line);
		std::istringstream iss1(line);
		while (iss1 >> temp){
			phiList.push_back(temp);
		}	
		counter = color*Mnum;

	} else if (mainL==7 && auxL==6){
		std::string pathName = "L76L7Samples.txt";
		std::ifstream myfile(pathName);
		std::string line;
		double temp;

		std::getline(myfile, line);
		std::istringstream iss(line);
		while (iss >> temp){
			readin.push_back(temp);
		}

		std::getline(myfile, line);
		std::istringstream iss1(line);
		while (iss1 >> temp){
			phiList.push_back(temp);
		}			
		counter = color*Mnum;

	} else if (mainL==6 && auxL==7){
		std::string pathName = "L76L6Samples.txt";
		std::ifstream myfile(pathName);
		std::string line;
		double temp;

		std::getline(myfile, line);
		std::istringstream iss(line);
		while (iss >> temp){
			readin.push_back(temp);
		}

		std::getline(myfile, line);
		std::istringstream iss1(line);
		while (iss1 >> temp){
			phiList.push_back(temp);
		} 
		counter = color*Mnum;		
	}
};

void stokesSolver::turnondummy(int mainL_, int auxL_, int M_, int color_){
	dummy=1;
	readData(mainL_, auxL_, M_, color_);
	std::cout << "dummy data read" << std::endl;
};

void stokesSolver::H1test(){
	double output;

	Vec load1;
	Vec load2;
	Vec reference;
	Vec solution;

	mesh_periodic mesh_L7(7);
	mesh_L7.M_matrix();
	mesh_L7.A_matrix();
	mesh_L7.load_vector(load1, ref, NULL); 
	VecDuplicate(load1, &reference);
	mesh_L7.interpolate(load1, reference);
	mesh_L7.load_vector(load2, getValues_static, this);
	VecDuplicate(load2, &solution); 
	mesh_L7.interpolate(load2, solution);

	Vec workspaceV;
	Mat workspaceA;
	Mat workspaceM;

	// VecView(solution, PETSC_VIEWER_STDOUT_WORLD);
	// VecView(reference, PETSC_VIEWER_STDOUT_WORLD);
	VecAXPY(solution, -1.0, reference);
	VecDuplicate(solution, &workspaceV);

	Mat listFunction[] = {mesh_L7.A, NULL, NULL, NULL, mesh_L7.A, NULL, NULL, NULL, mesh_L7.zeroMatrix};
	MatCreateNest(PETSC_COMM_SELF, 3, NULL, 3, NULL, listFunction, &workspaceA);	
	MatMult(workspaceA, solution, workspaceV);
	VecDot(solution, workspaceV, &output);
	output = sqrt(output);
	std::cout << "H1 norm: " << output << std::endl;
	
	Mat listFunction2[] = {mesh_L7.M, NULL, NULL, NULL, mesh_L7.M, NULL, NULL, NULL, mesh_L7.zeroMatrix};
	MatCreateNest(PETSC_COMM_SELF, 3, NULL, 3, NULL, listFunction2, &workspaceM);	
	MatMult(workspaceM, solution, workspaceV);
	VecDot(solution, workspaceV, &output);
	output = sqrt(output);
	std::cout << "L2 norm: " << output << std::endl;

	VecDestroy(&load1);
	VecDestroy(&load2);
	VecDestroy(&reference);
	VecDestroy(&solution);
	VecDestroy(&workspaceV);
	MatDestroy(&workspaceM);
	MatDestroy(&workspaceA);
};

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

	// // Options List 1
	// PetscOptionsSetValue(NULL, "-ksp_type", "fgmres");
	// PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_type", "gamg");

	// // Options List 2
	// PetscOptionsSetValue(NULL, "-ksp_type", "gmres");
	// PetscOptionsSetValue(NULL, "-fieldsplit_0_ksp_type", "preonly");
	// PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_type", "lu");
	// PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_factor_mat_solver_type", "mumps");
	// PetscOptionsSetValue(NULL, "-fieldsplit_1_ksp_type", "preonly");

	// Options List 2
	PetscOptionsSetValue(NULL, "-ksp_type", "gmres");
	PetscOptionsSetValue(NULL, "-pc_fieldsplit_type", "additive");
	PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_type", "gamg");
	PetscOptionsSetValue(NULL, "-fieldsplit_0_ksp_type", "preonly");
	PetscOptionsSetValue(NULL, "-fieldsplit_1_pc_type", "jacobi");
	PetscOptionsSetValue(NULL, "-fieldsplit_1_ksp_type", "preonly");

	// //Options List 3 direct solver mumps
	// PetscOptionsSetValue(NULL, "-ksp_type", "preonly");
	// PetscOptionsSetValue(NULL, "-pc_type", "lu");
	// PetscOptionsSetValue(NULL, "-pc_factor_mat_solver_type", "mumps");
	// PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "2");
	// PetscOptionsSetValue(NULL, "-mat_mumps_icntl_24", "1");
	// PetscOptionsSetValue(NULL, "-mat_mumps_cntl_3", "1e-6");

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
	int parallelChain = std::stoi(value[num_term+5]);
	int plainMCMC_sample_number = std::stoi(value[num_term+6]);
	int obsNumPerRow = std::stoi(value[num_term+7]);
	double noiseVariance = std::stod(value[num_term+8]);
	double randomSeed = std::stoi(value[num_term+9]);

	if (rank == 0){
		std::cout << "configuration: " << std::endl; 
		std::cout << "num_term: " << num_term << " coefs: ";
		for (int i = 0; i < num_term; ++i){
			std::cout << rand_coef[i] << " ";
		}
		std::cout << std::endl;
		std::cout << "levels: "           << levels                  <<  std::endl;
		std::cout << "a: "                << a                       <<  std::endl;
		std::cout << "pCNstep: "          << pCNstep                 <<  std::endl;
		std::cout << "task: "             << task                    <<  std::endl;
		std::cout << "parallelChain: "    << parallelChain           <<  std::endl;
		std::cout << "plainMCMC samples:" << plainMCMC_sample_number <<  std::endl;
		std::cout << "obsNumPerRow: "     << obsNumPerRow            <<  std::endl;
		std::cout << "noiseVariance: "    << noiseVariance           <<  std::endl; 
		std::cout << "randomSeed: "       << randomSeed              <<  std::endl;
	}

    if (task == 0){
		stokesSolver testsolver(levels, 1, 1);
        testsolver.samples[0]=0.8;
        testsolver.solve(0);
        std::cout << testsolver.obsOutput() << std::endl;
        std::cout << testsolver.qoiOutput() << std::endl;
    }

	if (task == 1){
		double output;
		levels = 0;

		// MLMCMC_Bi_Uniform<pCN<stokesSolver>, stokesSolver> MLMCMCSolver(PETSC_COMM_SELF, levels, 1, rank, a, noiseVariance, 1.0);
		// output = MLMCMCSolver.mlmcmcRun();

		MLMCMC_Bi<pCN<stokesSolver>, stokesSolver> MLMCMCSolver(PETSC_COMM_SELF, levels, 1, rank, a, noiseVariance, 1.0);
		output = MLMCMCSolver.mlmcmcRun();

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

	if (task == 2){
		double output;

		if (parallelChain == 1){
			MPI_Comm PSubComm;

			int color = rank % (size/levels);
			// int color = rank % (size/(levels+1)); //temprary
			MPI_Comm_split(PETSC_COMM_WORLD, color, rank/levels, &PSubComm);			
			// MPI_Comm_split(PETSC_COMM_WORLD, color, rank/(levels+1), &PSubComm); //temprary			

			// MLMCMC_Bi_Uniform<pCN<stokesSolver>, stokesSolver> MLMCMCSolver(PSubComm, levels, 1, color, a, noiseVariance, 1.0);
			// output = MLMCMCSolver.mlmcmcRun();

			MLMCMC_Bi<pCN<stokesSolver>, stokesSolver> MLMCMCSolver(PSubComm, levels, 1, (rank+randomSeed)*randomSeed, a, noiseVariance, 1.0);
			output = MLMCMCSolver.mlmcmcRun();

			if (rank == color){
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

			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 0){
				double buffer;
				std::string finaloutput = "finalOutput";
				std::ofstream outputfile;
				outputfile.open(finaloutput);
				for (int i = 0; i < size/levels; i++){
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

		} else {
			// MLMCMC_Bi_Uniform<pCN<stokesSolver>, stokesSolver> MLMCMCSolver(PETSC_COMM_SELF, levels, 1, rank*randomSeed/2, a, noiseVariance, 1.0);
			// output = MLMCMCSolver.mlmcmcRun();			

			MLMCMC_Bi<pCN<stokesSolver>, stokesSolver> MLMCMCSolver(PETSC_COMM_SELF, levels, 1, rank, a, noiseVariance, 1.0);
			output = MLMCMCSolver.mlmcmcRun();

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
	}

	if (task == 3){
		double QoIOutputL6;
		double QoIOutputL5;
		double array[2400];

		if (rank == 0){
			std::string pathName = "0chain.txt";
			std::ifstream myfile;
			myfile.open(pathName, std::ios_base::in);
			for (int j = 0; j < 2400; ++j){
				myfile >> array[j];
			}
			myfile.close();
		}

		double samples2brun[30];
		MPI_Scatter(&array[0], 30, MPI_DOUBLE, &samples2brun[0], 30, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		stokesSolver testSolverL6(6, 1, 1);
		stokesSolver testSolverL5(5, 1, 1);

		std::string L6outputfile = "0chainL6output_";
		L6outputfile.append(std::to_string(rank));

		std::string L5outputfile = "0chainL5output_";
		L5outputfile.append(std::to_string(rank));

		std::ofstream myfileL6;
		std::ofstream myfileL5;

		myfileL6.open(L6outputfile);
		myfileL5.open(L5outputfile);

		for (int i = 0; i < 30; ++i){
			testSolverL6.samples[0] = samples2brun[i];
			testSolverL5.samples[0] = samples2brun[i];

			testSolverL6.solve(1);
			testSolverL5.solve(1);			

			QoIOutputL6 = testSolverL6.solve4QoI();
			QoIOutputL5 = testSolverL5.solve4QoI();

			myfileL6 << samples2brun[i] << " " << QoIOutputL6;
			myfileL6 << std::endl;		

			myfileL5 << samples2brun[i] << " " << QoIOutputL5;
			myfileL5 << std::endl;	
		}
		myfileL6.close();
		myfileL5.close();			

		if (rank == 0){
			std::string pathName2 = "1chain.txt";
			std::ifstream myfile2;
			myfile2.open(pathName2, std::ios_base::in);
			for (int j = 0; j < 2400; ++j){
				myfile2 >> array[j];
			}
			myfile2.close();
		}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Scatter(&array[0], 30, MPI_DOUBLE, &samples2brun[0], 30, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		std::string L6outputfile2 = "1chainL6output_";
		L6outputfile2.append(std::to_string(rank));

		std::string L5outputfile2 = "1chainL5output_";
		L5outputfile2.append(std::to_string(rank));

		std::ofstream myfileL62;
		std::ofstream myfileL52;

		myfileL62.open(L6outputfile2);
		myfileL52.open(L5outputfile2);

		for (int i = 0; i < 30; ++i){
			testSolverL6.samples[0] = samples2brun[i];
			testSolverL5.samples[0] = samples2brun[i];

			testSolverL6.solve(1);
			testSolverL5.solve(1);			

			QoIOutputL6 = testSolverL6.solve4QoI();
			QoIOutputL5 = testSolverL5.solve4QoI();

			myfileL6 << samples2brun[i] << " " << QoIOutputL6;
			myfileL6 << std::endl;		

			myfileL5 << samples2brun[i] << " " << QoIOutputL5;
			myfileL5 << std::endl;	
		}
		myfileL6.close();
		myfileL5.close();			
	}

	if (task == 6){ //Generate Reference
		double quadx[256]; //= {-0.9894, -0.9446, -0.8656, -0.7554, -0.6179, -0.4580, -0.2816, -0.0950,   0.0950,    0.2816,    0.4580,    0.6179,   0.7554,  0.8656,    0.9446,    0.9894};
		double quadw[256]; //= {0.0272, 0.0623, 0.0952, 0.1246, 0.1496, 0.1692, 0.1826, 0.1895, 0.1895,    0.1826,    0.1692,    0.1496,    0.1246, 0.0952,    0.0623,    0.0272};
		
		double obs;
		double qoi;

		double localNum=0;
		double localDen=0;
 
		// gauleg(128, quadx, quadw);  //gaussian legendre
		gauher2(128, quadx, quadw);     //gaussian hermite

		stokesSolver refSolver(levels, 1, 1);
		// stokesSolver refSolver2(levels-1, 1, 1);

		for (int i = 0; i < 8; ++i){
			refSolver.samples[0] = sqrt(2.0)*quadx[rank+8*i];
			// refSolver2.samples[0] = sqrt(2.0)*quadx[rank+8*i];
			std::cout << "rank: " << rank << " coef: " << quadx[rank+8*i] << std::endl;
			refSolver.solve(0);
			// refSolver2.solve(0);

			obs = refSolver.obsOutput();
			// obs = obs*2-refSolver2.obsOutput();

			refSolver.solve4QoI();
			// refSolver2.solve4QoI();
			qoi = refSolver.qoiOutput();
			// qoi = qoi*2-refSolver2.qoiOutput();

			localNum += qoi*quadw[rank+8*i]*exp(-1.0/noiseVariance*pow(obs+0.965649652758160, 2)/2)+(-qoi)*quadw[rank+8*i]*exp(-1.0/noiseVariance*pow(-obs+0.965649652758160, 2)/2);
			localDen += quadw[rank+8*i]*exp(-1.0/noiseVariance*pow(obs+0.965649652758160, 2)/2)+quadw[rank+8*i]*exp(-1.0/noiseVariance*pow(-obs+0.965649652758160, 2)/2);
			// localNum += qoi*quadw[rank]*exp(-10*pow(obs+0.994214, 2)/2)+(-qoi)*quadw[rank]*exp(-10*pow(-obs+0.994214, 2)/2);
			// localDen += quadw[rank]*exp(-10*pow(obs+0.994214, 2)/2)+quadw[rank]*exp(-10*pow(-obs+0.994214, 2)/2);
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
		double ObsOutputL6;

		stokesSolver testSolverL6(levels, num_term, 1);
		testSolverL6.samples[0] = 1.0;
		// testSolverL6.samples[1] = 0.5;
		// testSolverL6.samples[2] = 0.3;
		// testSolverL6.samples[3] = 0.8;
		testSolverL6.solve(0);

		ObsOutputL6 = testSolverL6.obsOutput();

		std::cout << ObsOutputL6;
		std::cout << std::endl;		
	}

	PetscFinalize();
	return 0;
}

