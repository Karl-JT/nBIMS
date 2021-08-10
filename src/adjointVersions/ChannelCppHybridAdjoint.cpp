#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>

#include <adolc/adolc.h>
#include "LBFGS/include/LBFGS.h"
#include <Eigen/Core>
#include <petscksp.h>

#include "../forwardSolverArray/ChannelCppSolverArray.h"
#include "../forwardSolverArray/ChannelCppSolverArray.cpp"

using namespace std;
using Eigen::VectorXd;
using namespace LBFGSpp;

template<typename T>
double mainSolver(caseProp<T>& turbProp, vector<double>& adjointTerm){
	iterativeSolver(turbProp);
	//system("python CppoutputReader.py");
//////////////////////////////////////////////////////////////////	
	// ofstream yFile;
	// ofstream uFile;
	// ofstream betaFile;
	// ofstream kfile;
	// ofstream omegaFile;
	// ofstream nutFile;
	// yFile.open("./output/yFile.csv");
	// uFile.open("./output/uFile.csv");
	// betaFile.open("./output/betaFile.csv");
	// kfile.open("./output/kFile.csv");
	// omegaFile.open("./output/omegaFile.csv");
	// nutFile.open("./output/nutFile.csv");
	// for (int i = 0; i < turbProp.m; i++) {
	// 	yFile << turbProp.yCoordinate[i] << endl;
	// 	uFile << turbProp.xVelocity[i]<< endl;
	// 	betaFile << turbProp.betaML[i] << endl;
	// 	kfile << turbProp.k[i] << endl;
	// 	omegaFile << turbProp.omega[i] << endl;
	// 	nutFile << turbProp.nut[i] << endl;
	// }
	// yFile.close();
	// uFile.close();
	// betaFile.close();
	// kfile.close();
	// omegaFile.close();
	// nutFile.close();
////////////////////////////////////////////////////////////////////
	caseProp<adouble> adjointTurbProp;
	caseInitialize(adjointTurbProp, turbProp.reTau);
	initialization(adjointTurbProp);
	for (unsigned i = 0; i < turbProp.m; i++){
		adjointTurbProp.xVelocity[i] = turbProp.xVelocity[i];
		adjointTurbProp.k[i] = turbProp.k[i];
		adjointTurbProp.omega[i] = turbProp.omega[i];
		adjointTurbProp.nut[i] = turbProp.nut[i];
		adjointTurbProp.betaML[i] = turbProp.betaML[i];		
	}

	for (unsigned i = 0; i < 3*turbProp.m; i++){
		adjointTurbProp.R[i] = turbProp.R[i];
		adjointTurbProp.solution[i] = turbProp.solution[i];
	}

	double* wP = new double[3*turbProp.m];
	double* PR = new double[3*turbProp.m];

	trace_on(0);
	for (int i = 0; i < turbProp.m; i++){
		wP[i] = turbProp.xVelocity[i];
		wP[i+199] = turbProp.omega[i];
		wP[i+398] = turbProp.k[i];
	}	
	for (int i = 0; i < turbProp.m; i++){
		adjointTurbProp.xVelocity[i] <<= wP[i];
	}
	for (int i = 0; i < turbProp.m; i++){
		adjointTurbProp.omega[i] <<= wP[i+199];
	}
	for (int i = 0; i < turbProp.m; i++){
		adjointTurbProp.k[i] <<= wP[i+398];
	}
	residualUpdate(adjointTurbProp);
	for (unsigned i = 0; i < 597; i++){
		adjointTurbProp.R[i] >>= PR[i];
	}
	trace_off(0);
	double* PRPW[3*turbProp.m];
	for (int i = 0; i < 3*turbProp.m; i++){
		PRPW[i] = new double[3*turbProp.m];
	}
	jacobian(0, 3*turbProp.m, 3*turbProp.m, wP, PRPW);

	ofstream PRPWfile;
	PRPWfile.open("./output/PRPW.csv");
	for (int i = 0; i < 3*turbProp.m; i++) {
		for (int j = 0; j < 3*turbProp.m; j++){
			PRPWfile << PRPW[i][j] << " ";
		}
		PRPWfile << endl;
	}
	PRPWfile.close();

	vector<double> PJPU(597, 0);
	for (unsigned i = 0; i < 199; i++){
		PJPU[i] = 2e6*(turbProp.xVelocity[i] - turbProp.dnsData[i]*turbProp.frictionVelocity);
	}
	ofstream PJPUFile;
	PJPUFile.open("./output/PJPU.csv"); 
	for (int i = 0; i < 3*turbProp.m; i++) {
		PJPUFile << PJPU[i] << endl;
	}
	PJPUFile.close();

	vector<double> PRPbeta(199, 0);
	PRPbeta[0] = 0;//5.0 / 9.0 * pow(((turbProp.xVelocity[1] - turbProp.xVelocity[0]) / (turbProp.yCoordinate[1] - turbProp.yCoordinate[0])), 2.0);
	PRPbeta[1] = 0;
	PRPbeta[2] = 0;
	for (unsigned i = 3; i < 196; i++){
		PRPbeta[i] = 5.0 / 9.0 * pow(((turbProp.xVelocity[i + 1] - turbProp.xVelocity[i - 1]) / (turbProp.yCoordinate[i + 1] - turbProp.yCoordinate[i - 1])), 2.0);
	}
	PRPbeta[196] = 0;
	PRPbeta[197] = 0;
	PRPbeta[198] = 0;//5.0 / 9.0 * pow(((turbProp.xVelocity[198] - turbProp.xVelocity[197]) / (turbProp.yCoordinate[198] - turbProp.yCoordinate[197])), 2.0);
	ofstream PRPbetaFile;
	PRPbetaFile.open("./output/PRPbeta.csv");
	for (unsigned i = 0; i < 199; i++){
		PRPbetaFile << PRPbeta[i] << endl;
	}
	PRPbetaFile.close();

	vector<vector<double>> PRPWTranspose(3*turbProp.m);
	for (unsigned i = 0; i < 3*turbProp.m; i++){
		PRPWTranspose[i].resize(3*turbProp.m);
		for (unsigned j = 0; j < 3*turbProp.m; j++){
			PRPWTranspose[i][j] = PRPW[j][i];
		}
	}

	vector<vector<double>> A(597);
	A = PRPWTranspose;
	A.erase(A.begin()+596);
	A.erase(A.begin()+395, A.begin()+399);
	A.erase(A.begin()+198, A.begin()+202);
	A.erase(A.begin());
	for (unsigned i = 0; i < 587; i++){
		A[i].erase(A[i].begin()+596);
		A[i].erase(A[i].begin()+395, A[i].begin()+399);
		A[i].erase(A[i].begin()+198, A[i].begin()+202);
		A[i].erase(A[i].begin());
	}

	vector<double> bMatrix(587);
	for (unsigned i = 0; i < 587; i++){
		bMatrix[i] = PJPU[i+1];
	}

	//********************************************************************************//
	//***********************adjoint solver part**************************************//
	Vec x, b;
	Mat aMatrix;
	KSP ksp;
	PC pc; 
	PetscReal norm;
	PetscErrorCode ierr;
	PetscInt n = 587, col[587], its=587, restart = 587;
	PetscMPIInt size = 1;
	PetscScalar value[587];
	PetscScalar *matrix;
	PetscScalar *array;

	PetscInitialize(NULL,NULL,(char*)0,NULL); 
	MPI_Comm_size(PETSC_COMM_WORLD,&size);
	//ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);//CHKERRQ(ierr);

	VecCreate(PETSC_COMM_WORLD,&x);
	PetscObjectSetName((PetscObject) x, "Solution");
	VecSetSizes(x,PETSC_DECIDE,n);
	VecSetFromOptions(x);
	VecDuplicate(x,&b);

	MatCreate(PETSC_COMM_WORLD,&aMatrix);
	MatSetSizes(aMatrix,PETSC_DECIDE,PETSC_DECIDE,n,n);
	MatSetFromOptions(aMatrix);
	MatSetUp(aMatrix);

	for (int i=0; i<587; i++) {
		for (int j=0; j<587; j++){
			value[j] = A[i][j];
			col[j] = j;
		}
		MatSetValues(aMatrix,1,&i,587,col,value,INSERT_VALUES);
	}
	MatAssemblyBegin(aMatrix,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(aMatrix,MAT_FINAL_ASSEMBLY);
	//MatView(aMatrix, PETSC_VIEWER_STDOUT_WORLD);

	for (int i=0; i<587; i++){
		value[i] = bMatrix[i];
		col[i] = i;
	}
	VecSetValues(b, 587, col, value, INSERT_VALUES);
	//VecView(b, PETSC_VIEWER_STDOUT_WORLD);

	KSPCreate(PETSC_COMM_WORLD,&ksp);//CHKERRQ(ierr);
	KSPSetOperators(ksp,aMatrix,aMatrix);//CHKERRQ(ierr);

	KSPGetPC(ksp,&pc);//CHKERRQ(ierr);
	PCSetType(pc,PCJACOBI);//CHKERRQ(ierr);
	KSPSetTolerances(ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);//CHKERRQ(ierr);
	KSPSetFromOptions(ksp);//CHKERRQ(ierr);
	KSPGMRESSetRestart(ksp, restart);

	KSPSolve(ksp,b,x);//CHKERRQ(ierr);

	vector<double> phi(199, 0);
	VecGetArray(x, &array);
	phi[0] = 0;
	phi[1] = 0;
	phi[2] = 0;
	phi[196] = 0;
	phi[197] = 0;
	phi[198] = 0;
	for (int i = 0; i < 193; i++){
		phi[i+3] = array[i+197];
	}

	adjointTerm[0] = 0;
	adjointTerm[1] = 0;
	adjointTerm[2] = 0;
	for (int i = 3; i < 196; i++){
		adjointTerm[i] = phi[i]*PRPbeta[i];
	} 
	adjointTerm[196] = 0;
	adjointTerm[197] = 0;
	adjointTerm[198] = 0;
	//KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);

	//ierr = KSPGetIterationNumber(ksp,&its);//CHKERRQ(ierr);
	//ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);//CHKERRQ(ierr);

	ierr = VecDestroy(&x);//CHKERRQ(ierr); 
	ierr = VecDestroy(&b);//CHKERRQ(ierr); 
	ierr = MatDestroy(&aMatrix);//CHKERRQ(ierr);
	ierr = KSPDestroy(&ksp);//CHKERRQ(ierr);

	//PetscFinalize();
	double J = 0;
	for (int i = 1; i < turbProp.m-1; i++) {
		J = J + 1e6*pow((turbProp.xVelocity[i]-turbProp.dnsData[i]*turbProp.frictionVelocity), 2) + 4*pow((turbProp.betaML[i]-1), 2);
	}
	return J;
};

class optLoop{
private:
	int n;
	caseProp<double> turbProp; 
public:
	optLoop(int n_) : n(n_) {
		caseInitialize(turbProp, 550);
		initialization(turbProp);
	}
	~optLoop(){};

	double operator()(const VectorXd& x, VectorXd& grad){
		vector<double> PJPbeta(n, 0);
		double J = 0;
		vector<double> adjointTerm(199, 0);

		cout << endl << "betaValue: "<< endl;
		for (int i = 0; i < n; i++){
			turbProp.betaML[i] = x[i];
			cout << turbProp.betaML[i] << " ";
		}
		J = mainSolver(turbProp, adjointTerm);
		for (int i = 0; i < turbProp.m; i++){
			PJPbeta[i] = 2*4*(turbProp.betaML[i]-1);
		}		
		cout << endl << "gradient" << endl;
		for (int i = 0; i < n; i++){
			grad[i] = PJPbeta[i] - adjointTerm[i];
			cout << grad[i] << " ";
		}
		// if (turbProp.optIter % 100 == 0){
		// 	system("python CppoutputReader.py");
		// }
		//turbProp.optIter++;
		ofstream costFile;
		costFile.open("./output/costFunction.csv", ofstream::app);
		costFile << J << endl;
		cout << endl << endl << "Object Function: " << J << endl;
		return J;
	}
};

int main(int argc, char **argv){
	//double reTau = stof(argv[1]);
	const int n = 199;

	LBFGSParam<double> param;
	param.epsilon = 0.00001;
	param.max_iterations = 150;

	LBFGSSolver<double> solver(param);
	optLoop optStep(n);

	VectorXd x =  VectorXd::Zero(n);
	for (int i = 0; i < n; i++){
		x[i] = 1.0;
	}
	double fx;
	int niter = solver.minimize(optStep, x, fx);

	cout << niter << " iterations" << endl;
	system("python CppoutputReader.py");

	return 0;
}