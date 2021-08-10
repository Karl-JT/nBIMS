#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>

#include <adolc/adolc.h>
#include "LBFGS/include/LBFGS.h"
#include <Eigen/Core>
#include <petscksp.h>

#include "dnsDataInterpreter.H"

using namespace std;
using Eigen::VectorXd;
using namespace LBFGSpp;

class caseProp {
public:
	int m, simpleGridingRatio, deltaTime;
	double nu, frictionVelocity;
};
caseProp updateCaseProperties(int reTau) {
	caseProp turbProp;
	if (reTau == 180) {
		turbProp.m = 100;
		turbProp.simpleGridingRatio = 100;
		turbProp.deltaTime = 1;
		turbProp.nu = 3.4e-4;
		turbProp.frictionVelocity = 6.37309e-2;
	}
	if (reTau == 550) {
		turbProp.m = 100;
		turbProp.simpleGridingRatio = 150;
		turbProp.deltaTime = 100;
		turbProp.nu = 1e-4;
		turbProp.frictionVelocity = 5.43496e-2;
	}
	if (reTau == 1000) {
		turbProp.m = 100;
		turbProp.simpleGridingRatio = 1000;
		turbProp.deltaTime = 1;
		turbProp.nu = 5e-5;
		turbProp.frictionVelocity = 5.00256e-2;
	}
	if (reTau == 2000) {
		turbProp.m = 100;
		turbProp.simpleGridingRatio = 2500;
		turbProp.deltaTime = 1;
		turbProp.nu = 2.3e-5;
		turbProp.frictionVelocity = 4.58794e-2;
	}
	if (reTau == 5200) {
		turbProp.m = 100;
		turbProp.simpleGridingRatio = 2000;
		turbProp.deltaTime = 1;
		turbProp.nu = 8e-6;
		turbProp.frictionVelocity =4.14872e-2;
	}
	return turbProp;
};
template<typename T>
void flowSolver(caseProp& turbProp, vector<double>& yCoordinate, vector<T>& xVelocity, vector<T>& nut) {
	vector<T> vectorA(turbProp.m, 0);
	vector<T> vectorB(turbProp.m, 0);
	vector<T> vectorC(turbProp.m, 0);
	vector<T> vectorD(turbProp.m, 1);
	

	vectorB[0] = -1;
	vectorC[0] = 0;
	vectorA[turbProp.m - 1] = 1;
	vectorB[turbProp.m - 1] = -1;
	vectorD[0] = 0;
	vectorD[turbProp.m - 1] = 0;

	for (int i = 1; i < turbProp.m - 1; i++) {
		vectorA[i] = (2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(turbProp.nu + nut[i - 1] / 2.0 + nut[i] / 2.0) / (yCoordinate[i] - yCoordinate[i - 1]))*turbProp.deltaTime;
		vectorB[i] = (-2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*((turbProp.nu + nut[i] / 2.0 + nut[i + 1] / 2.0) / (yCoordinate[i + 1] - yCoordinate[i]) + (turbProp.nu + nut[i - 1] / 2.0 + nut[i] / 2.0) / (yCoordinate[i] - yCoordinate[i - 1])))*turbProp.deltaTime - 1;
		vectorC[i] = (2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(turbProp.nu + nut[i] / 2.0 + nut[i + 1] / 2.0) / (yCoordinate[i + 1] - yCoordinate[i]))*turbProp.deltaTime;
		vectorD[i] = vectorD[i] * (-pow(turbProp.frictionVelocity, 2)) * turbProp.deltaTime - xVelocity[i];
	}

	vector<T> newVectorC(turbProp.m, 0);
	vector<T> newVectorD(turbProp.m, 0);

	newVectorC[0] = vectorC[0] / vectorB[0];
	newVectorD[0] = vectorD[0] / vectorB[0];

	for (int i = 1; i < turbProp.m - 1; i++) {
		newVectorC[i] = vectorC[i] / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
		newVectorD[i] = (vectorD[i] - vectorA[i] * newVectorD[i - 1]) / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
	}
	newVectorD[turbProp.m - 1] = (vectorD[turbProp.m - 1] - vectorA[turbProp.m - 1] * newVectorD[turbProp.m - 2]) / (vectorB[turbProp.m - 1] - vectorA[turbProp.m - 1] * newVectorC[turbProp.m - 2]);

	xVelocity[turbProp.m - 1] = newVectorD[turbProp.m - 1];
	for (int i = 0; i < turbProp.m - 1; i++) {
		xVelocity[turbProp.m - 2 - i] = newVectorD[turbProp.m - 2 - i] - newVectorC[turbProp.m - 2 - i] * xVelocity[turbProp.m - 1 - i];
	}
};

template<typename T>
void omegaSolver(caseProp& turbProp, vector<double>& yCoordinate, vector<T>& xVelocity, vector<T>& nut, vector<T>& omega, vector<T>& betaML) {
	double sigma = 0.5;
	double alpha = 3.0 / 40.0;
	double gamma = 5.0 / 9.0;
	int boundaryPoints = 6;

	vector<T> vectorA(turbProp.m, 0);
	vector<T> vectorB(turbProp.m, 0);
	vector<T> vectorC(turbProp.m, 0);
	vector<T> vectorD(turbProp.m, 0);

	vectorB[0] = -1;
	vectorC[0] = 0;
	vectorD[0] = -1e50;

	for (int i = 1; i < boundaryPoints; i++) {
		vectorA[i] = 0;
		vectorB[i] = -1;
		vectorC[i] = 0;
		vectorD[i] = -6.0 * turbProp.nu / (0.00708 * pow(yCoordinate[i], 2));
	}

	vectorA[turbProp.m - 1] = 1;
	vectorB[turbProp.m - 1] = -1;
	vectorD[turbProp.m - 1] = 0;

	for (int i = boundaryPoints; i < turbProp.m - 1; i++) {
		vectorA[i] = 2.0 * turbProp.deltaTime / (yCoordinate[i + 1] - yCoordinate[i - 1])*(turbProp.nu + sigma * (nut[i] / 2.0 + nut[i - 1] / 2.0)) / (yCoordinate[i] - yCoordinate[i - 1]);
		vectorB[i] = turbProp.deltaTime * (-alpha * omega[i] - 2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*((turbProp.nu + sigma * (nut[i] / 2.0 + nut[i - 1] / 2.0)) / (yCoordinate[i] - yCoordinate[i - 1]) + (turbProp.nu + sigma * (nut[i] / 2.0 + nut[i + 1] / 2.0)) / (yCoordinate[i + 1] - yCoordinate[i]))) - 1;
		vectorC[i] = 2.0 * turbProp.deltaTime / (yCoordinate[i + 1] - yCoordinate[i - 1])*(turbProp.nu + sigma * (nut[i] / 2.0 + nut[i + 1] / 2.0)) / (yCoordinate[i + 1] - yCoordinate[i]);
		vectorD[i] = (-gamma * betaML[i] * pow(((xVelocity[i + 1] - xVelocity[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1])), 2.0))*turbProp.deltaTime - omega[i]; // - max(0, sigmaD / omega[i] * ((k[i + 1] - k[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]))*((omega[i + 1] - omega[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]))))*deltaTime - omega[i];
	}

	vector<T> newVectorC(turbProp.m, 0);
	vector<T> newVectorD(turbProp.m, 0);

	newVectorC[0] = vectorC[0] / vectorB[0];
	newVectorD[0] = vectorD[0] / vectorB[0];
	for (int i = 1; i < turbProp.m - 1; i++) {
		newVectorC[i] = vectorC[i] / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
		newVectorD[i] = (vectorD[i] - vectorA[i] * newVectorD[i - 1]) / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
	}
	newVectorD[turbProp.m - 1] = (vectorD[turbProp.m - 1] - vectorA[turbProp.m - 1] * newVectorD[turbProp.m - 2]) / (vectorB[turbProp.m - 1] - vectorA[turbProp.m - 1] * newVectorC[turbProp.m - 2]);

	omega[turbProp.m - 1] = newVectorD[turbProp.m - 1];
	for (int i = 0; i < turbProp.m - 1; i++) {
		omega[turbProp.m - 2 - i] = newVectorD[turbProp.m - 2 - i] - newVectorC[turbProp.m - 2 - i] * omega[turbProp.m - 1 - i];
	}
};

template<typename T>
void kSolver(caseProp& turbProp, vector<double>& yCoordinate, vector<T>& xVelocity, vector<T>& nut, vector<T>& omega, vector<T>& k) {
	double sigmaStar = 0.5;
	double alphaStar = 0.09;

	vector<T> vectorA(turbProp.m, 0);
	vector<T> vectorB(turbProp.m, 0);
	vector<T> vectorC(turbProp.m, 0);
	vector<T> vectorD(turbProp.m, 0);

	vectorB[0] = -1.0;
	vectorC[0] = 0.0;
	vectorA[turbProp.m - 1] = 1.0;
	vectorB[turbProp.m - 1] = -1.0;
	vectorD[0] = 0;
	vectorD[turbProp.m - 1] = 0;

	for (int i = 1; i < turbProp.m - 1; i++){
		vectorA[i] = 2.0 * turbProp.deltaTime / (yCoordinate[i + 1] - yCoordinate[i - 1])*(turbProp.nu + sigmaStar * (nut[i] / 2 + nut[i - 1] / 2)) / (yCoordinate[i] - yCoordinate[i - 1]);
		vectorB[i] = turbProp.deltaTime * (-alphaStar * omega[i] - 2 / (yCoordinate[i + 1] - yCoordinate[i - 1])*((turbProp.nu + sigmaStar * (nut[i] / 2 + nut[i - 1] / 2)) / (yCoordinate[i] - yCoordinate[i - 1]) + (turbProp.nu + sigmaStar * (nut[i] / 2 + nut[i + 1] / 2)) / (yCoordinate[i + 1] - yCoordinate[i]))) - 1;
		vectorC[i] = 2.0 * turbProp.deltaTime / (yCoordinate[i + 1] - yCoordinate[i - 1])*(turbProp.nu + sigmaStar * (nut[i] / 2 + nut[i + 1] / 2)) / (yCoordinate[i + 1] - yCoordinate[i]);
		vectorD[i] = (-nut[i] * pow((xVelocity[i + 1] - xVelocity[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]), 2.0))*turbProp.deltaTime - k[i];
	}
	vector<T> newVectorC(turbProp.m);
	vector<T> newVectorD(turbProp.m);

	newVectorC[0] = vectorC[0] / vectorB[0];
	newVectorD[1] = vectorD[0] / vectorB[0];
	for (int i = 1; i < turbProp.m - 1; i++) {
		newVectorC[i] = vectorC[i] / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
		newVectorD[i] = (vectorD[i] - vectorA[i] * newVectorD[i - 1]) / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
	}
	newVectorD[turbProp.m - 1] = (vectorD[turbProp.m - 1] - vectorA[turbProp.m - 1] * newVectorD[turbProp.m - 2]) / (vectorB[turbProp.m - 1] - vectorA[turbProp.m - 1] * newVectorC[turbProp.m - 2]);

	k[turbProp.m - 1] = newVectorD[turbProp.m - 1];
	for (int i = 0; i < turbProp.m - 1; i++) {
		k[turbProp.m - 2 - i] = newVectorD[turbProp.m - 2 - i] - newVectorC[turbProp.m - 2 - i] * k[turbProp.m - 1 - i];
	}
};
void mainSolver(vector<double>& betaML, vector<double>& R, vector<double>& PJPbeta, vector<double>& phi, double& J, vector<vector<double>>& PRPbetaTape){
	// Parameters
	double reTau = 550;
	caseProp turbProp = updateCaseProperties(reTau);

	// Node Coordinate
	vector<double> yCoordinate(turbProp.m);

	// Variables on Nodes
	vector<double> xVelocity(turbProp.m, 0);
	vector<double> k(turbProp.m, 1e-8);
	vector<double> omega(turbProp.m, 1e5);
	vector<double> nut(turbProp.m, 1e-5);

	// Node Coordinate Initialization
	double gridRatio = pow(turbProp.simpleGridingRatio, 1.0 / (turbProp.m - 2));
	double firstNodeDist = (1 - gridRatio) / (1 - pow(gridRatio, turbProp.m - 1));

	double tempGridSize = firstNodeDist;
	yCoordinate[0] = 0;
	for (int i = 1; i < turbProp.m; i++) {
		yCoordinate[i] = yCoordinate[i - 1] + tempGridSize;
		tempGridSize = tempGridSize * gridRatio;
	}
	dnsDataInterpreter dnsData(yCoordinate, reTau);
	vector<double> xVelocityTemp;
	for (int i = 0; i < 1000000; i++) {
		//cout << "#";
		xVelocityTemp = xVelocity;
		flowSolver(turbProp, yCoordinate, xVelocity, nut);
		omegaSolver(turbProp, yCoordinate, xVelocity, nut, omega, betaML);
		kSolver(turbProp, yCoordinate, xVelocity, nut, omega, k);
		adouble res=0.0;
		for (int i = 0; i < turbProp.m; i++) {
			nut[i] = k[i] / omega[i];
			R[i] = abs(xVelocity[i] - xVelocityTemp[i]);
			res = res + R[i];
		}
		if (res < 1e-8){
			cout << endl << "res: " << res << endl;
			break;
		}
	}

	ofstream yFile;
	ofstream uFile;
	ofstream betaFile;
	yFile.open("yFile.csv");
	uFile.open("uFile.csv");
	betaFile.open("betaFile.csv");
	for (int i = 0; i < yCoordinate.size(); i++) {
		yFile << yCoordinate[i] << endl;
		uFile << xVelocity[i]<< endl;
		betaFile << betaML[i] << endl;
	}

	double* betaP = new double[turbProp.m];
	double* xVelocityP = new double[turbProp.m];
	double* Py = new double[turbProp.m];

	vector<adouble> tapeBetaML(turbProp.m);
	vector<adouble> tapexVelocity(turbProp.m);
	vector<adouble> residual(turbProp.m);
	vector<adouble> tapeNut(turbProp.m);
	vector<adouble> tapeOmega(turbProp.m);
	vector<adouble> tapeK(turbProp.m);

	trace_on(0);
	for (int i = 0; i < turbProp.m; i++){
		tapeBetaML[i] = betaML[i];
		xVelocityP[i] = xVelocity[i];
		tapeNut[i] = nut[i];
		tapeOmega[i] = omega[i];
		tapeK[i] = k[i];
	}
	for (int i = 0; i < turbProp.m; i++){
		tapexVelocity[i] <<= xVelocityP[i];
	}

	omegaSolver(turbProp, yCoordinate, tapexVelocity, tapeNut, tapeOmega, tapeBetaML);
	kSolver(turbProp, yCoordinate, tapexVelocity, tapeNut, tapeOmega, tapeK);
	for (int i = 0; i < turbProp.m; i++) {
		tapeNut[i] = tapeK[i] / tapeOmega[i];
	}
	flowSolver(turbProp, yCoordinate, tapexVelocity, tapeNut);

	residual[0] = 0;
	for (int i = 1; i < turbProp.m; i++){
		residual[i] = (tapexVelocity[i] - xVelocity[i])*(yCoordinate[i]-yCoordinate[i-1]);
	}			
	for (int i = 0; i < turbProp.m; i++){
		residual[i] >>= Py[i];
	}
	trace_off(0);
	double* PRPU[turbProp.m];
	for (int i = 0; i < turbProp.m; i ++){
		PRPU[i] = new double[turbProp.m];
	}
	jacobian(0, turbProp.m, turbProp.m, xVelocityP, PRPU);

	trace_on(1);
	for (int i = 0; i < turbProp.m; i++){
		betaP[i] = betaML[i];
		tapexVelocity[i] = xVelocity[i];
		tapeNut[i] = nut[i];
		tapeOmega[i] = omega[i];
		tapeK[i] = k[i];
	}
	for (int i = 0; i < turbProp.m; i++){
		tapeBetaML[i] <<= betaP[i];
	}

	omegaSolver(turbProp, yCoordinate, tapexVelocity, tapeNut, tapeOmega, tapeBetaML);
	kSolver(turbProp, yCoordinate, tapexVelocity, tapeNut, tapeOmega, tapeK);
	for (int i = 0; i < turbProp.m; i++) {
		tapeNut[i] = tapeK[i] / tapeOmega[i];
	}
	flowSolver(turbProp, yCoordinate, tapexVelocity, tapeNut);

	residual[0] = 0;
	for (int i = 1; i < turbProp.m; i++){
		residual[i] = (tapexVelocity[i] - xVelocity[i])*(yCoordinate[i]-yCoordinate[i-1]);
	}			
	for (int i = 0; i < turbProp.m; i++){
		residual[i] >>= Py[i];
	}
	trace_off(1);

	double* PRPbeta[turbProp.m];
	for (int i = 0; i < turbProp.m; i ++){
		PRPbeta[i] = new double[turbProp.m];
	}
	jacobian(1, turbProp.m, turbProp.m, betaP, PRPbeta);


	for (int i = 0; i < turbProp.m; i++){
		for (int j = 0; j < turbProp.m; j++){
			PRPbetaTape[i][j] = PRPbeta[i][j];
		}
	}

	vector<double> PJPU(turbProp.m, 0);
	J = 0;
	PJPbeta[0] = 0;
	PJPU[0] = 0;
	for (int i = 0; i < turbProp.m; i++){
		PRPU[0][i] = 0;
	}
	phi[0] = 0;

	for (int i = 1; i < turbProp.m; i++) {
		J = J + 1e20*pow((xVelocity[i]-dnsData.U[i]*turbProp.frictionVelocity)*(yCoordinate[i]-yCoordinate[i-1]), 2) + 1*pow((betaML[i]-1)*(yCoordinate[i]-yCoordinate[i-1]), 2);
		PJPbeta[i] = 2*1*(betaML[i]-1)*(yCoordinate[i]-yCoordinate[i-1]);
		PJPU[i] = 2*1e20*(xVelocity[i]-dnsData.U[i]*turbProp.frictionVelocity)*(yCoordinate[i]-yCoordinate[i-1]);
	}

	//********************************************************************************//
	//***********************adjoint solver part**************************************//
	Vec x, b;
	Mat A;
	KSP ksp;
	PC pc; 
	PetscReal norm;
	PetscErrorCode ierr;
	PetscInt n = 100, col[100], its;
	PetscMPIInt size = 1;
	PetscScalar value[100];
	PetscScalar *matrix;
	PetscScalar *array;

	ierr = PetscInitialize(NULL,NULL,(char*)0,NULL); //if (ierr) return ierr;
	ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);//CHKERRQ(ierr);
	//if (size != 1) SETERRQ(PETSC_COMM_WORLD,1,"This is a uniprocessor example only!");
	ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);//CHKERRQ(ierr);

	ierr = VecCreate(PETSC_COMM_WORLD,&x);//CHKERRQ(ierr);
	ierr = PetscObjectSetName((PetscObject) x, "Solution");//CHKERRQ(ierr);
	ierr = VecSetSizes(x,PETSC_DECIDE,n);//CHKERRQ(ierr);
	ierr = VecSetFromOptions(x);//CHKERRQ(ierr);
	ierr = VecDuplicate(x,&b);//CHKERRQ(ierr);

	ierr = MatCreate(PETSC_COMM_WORLD,&A);//CHKERRQ(ierr);
	ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);//CHKERRQ(ierr);
	ierr = MatSetFromOptions(A);//CHKERRQ(ierr);
	ierr = MatSetUp(A);//CHKERRQ(ierr);

	for (int i=0; i<100; i++) {
		for (int j=0; j<100; j++){
			value[j] = PRPU[j][i];
			col[j] = j;
		}
		ierr   = MatSetValues(A,1,&i,100,col,value,INSERT_VALUES);//CHKERRQ(ierr);
	}
	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);//CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);//CHKERRQ(ierr);

	for (int i=0; i<100; i++){
		value[i] = -PJPU[i];
	}
	ierr = VecSetValues(b, 100, col, value, INSERT_VALUES);//CHKERRQ(ierr);

	ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);//CHKERRQ(ierr);
	ierr = KSPSetOperators(ksp,A,A);//CHKERRQ(ierr);

	ierr = KSPGetPC(ksp,&pc);//CHKERRQ(ierr);
	ierr = PCSetType(pc,PCJACOBI);//CHKERRQ(ierr);
	ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);//CHKERRQ(ierr);

	ierr = KSPSetFromOptions(ksp);//CHKERRQ(ierr);

	ierr = KSPSolve(ksp,b,x);//CHKERRQ(ierr);

	cout << "adjoint variables: " << endl;
	VecGetArray(x, &array);
	for (int i = 0; i < 100; i++){
		phi[i] = array[i];
		cout << phi[i] << " ";
	}

	ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);//CHKERRQ(ierr);

	ierr = KSPGetIterationNumber(ksp,&its);//CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);//CHKERRQ(ierr);

	ierr = VecDestroy(&x);//CHKERRQ(ierr); 
	ierr = VecDestroy(&b);//CHKERRQ(ierr); 
	ierr = MatDestroy(&A);//CHKERRQ(ierr);
	ierr = KSPDestroy(&ksp);//CHKERRQ(ierr);

	//ierr = PetscFinalize();
	//********************************************************************************//

	cout << "Object Function: " << J << endl;
};

class optLoop{
private:
	int n;
public:
	optLoop(int n_) : n(n_) {}
	double operator()(const VectorXd& x, VectorXd& grad){
		vector<double> PJPbeta(n, 0);
		vector<vector<double>> PRPbetaTape(n);
		for (int i=0; i<n; i++){
			PRPbetaTape[i].resize(n);
		}	
		vector<double> phi(n, 0);
		vector<double> betaML(n);
		vector<double> R(n);
		double J = 0;

		for (int i = 0; i < n; i++){
			betaML[i] = x[i];
		}
		mainSolver(betaML, R, PJPbeta, phi, J, PRPbetaTape);
		cout << endl << "gradient" << endl;
		for (int i = 0; i < n; i++){
			grad[i] = PJPbeta[i];
			for (int j = 0; j < n; j++){
				grad[i] = grad[i] + phi[j]*PRPbetaTape[j][i];
			}
			cout << grad[i] << " ";
		}
		return J;
	}
};

int main(int argc, char **args){
	const int n = 100;

	LBFGSParam<double> param;
	param.epsilon = 1e-10;
	param.max_iterations = 100;

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