#include "turbSolverAdjoint.h"

channelFlowAdjoint::channelFlowAdjoint(int level){
	caseInitialize(level, 550.0);
	initialization();

	//Set up for petsc adjoint solver
	size = 1;
	restart = 3*m;

	SlepcInitialize(NULL, NULL, NULL, NULL);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	MatCreate(PETSC_COMM_WORLD, &PRPWMat);
	MatCreate(PETSC_COMM_WORLD, &PRPBetaMat);
	VecCreate(PETSC_COMM_WORLD, &PJPWVec);
	VecCreate(PETSC_COMM_WORLD, &adjoints);
	VecCreate(PETSC_COMM_WORLD, &adjointTerm);
	VecCreate(PETSC_COMM_WORLD, &PJPBetaVec);

	MatSetSizes(PRPWMat, PETSC_DECIDE, PETSC_DECIDE, 3*m, 3*m);
	MatSetSizes(PRPBetaMat, PETSC_DECIDE, PETSC_DECIDE, 3*m, m);
	VecSetSizes(PJPWVec, PETSC_DECIDE, 3*m);
	VecSetSizes(adjoints, PETSC_DECIDE, 3*m);
	VecSetSizes(adjointTerm, PETSC_DECIDE, m);
	VecSetSizes(PJPBetaVec, PETSC_DECIDE, m);

	MatSetFromOptions(PRPWMat);
	MatSetFromOptions(PRPBetaMat);
	VecSetFromOptions(PJPWVec);
	VecSetFromOptions(adjoints);
	VecSetFromOptions(adjointTerm);
	VecSetFromOptions(PJPBetaVec);

	MatSetUp(PRPWMat);
	MatSetUp(PRPBetaMat);
	VecSetUp(PJPWVec);
	VecSetUp(adjoints);
	VecSetUp(adjointTerm);
	VecSetUp(PJPBetaVec);

	KSPCreate(PETSC_COMM_WORLD,&ksp);

	Mat obsMat;
	Vec inputSpace, workSpace1, workSpace2;
	MatCreate(PETSC_COMM_WORLD, &obsMat);
	VecCreate(PETSC_COMM_WORLD, &inputSpace);
	VecCreate(PETSC_COMM_WORLD, &workSpace1);
	VecCreate(PETSC_COMM_WORLD, &workSpace2);
	MatSetSizes(obsMat, PETSC_DECIDE, PETSC_DECIDE, 3*m, 3*m);
	VecSetSizes(inputSpace, PETSC_DECIDE, m);
	VecSetSizes(workSpace1, PETSC_DECIDE, 3*m);
	VecSetSizes(workSpace2, PETSC_DECIDE, 3*m);
	MatSetFromOptions(obsMat);
	VecSetFromOptions(inputSpace);
	VecSetFromOptions(workSpace1);
	VecSetFromOptions(workSpace2);
	MatSetUp(obsMat);
	VecSetUp(inputSpace);
	VecSetUp(workSpace1);
	VecSetUp(workSpace2);
}

void channelFlowAdjoint::soluFwd(){
	iterativeSolver();
}

double channelFlowAdjoint::lnLikelihood(double sample[], int numCoef){
	int idx = 0;
	double lnLikelihoodt1 = 0;
	double u = 0;

	for (int i = 0; i < 577; i++){
		while (validationYCoordinate[i] > yCoordinate[idx+1]){
			idx++;
		}
		u = xVelocity[idx+1].value() - (xVelocity[idx+1].value() - xVelocity[idx].value())/(yCoordinate[idx+1] - yCoordinate[idx])*(yCoordinate[idx+1] - validationYCoordinate[i]);
		lnLikelihoodt1 = lnLikelihoodt1 - 1e6*pow(u-validationVelocity[i], 2);
	}

	for (int i = 0; i < numCoef; i++){
		lnLikelihoodt1 = lnLikelihoodt1 - pow(sample[i], 2);
	}

	return lnLikelihoodt1;
}

void channelFlowAdjoint::updateParameter(double sample[], int numCoef){
	for (int i = 0; i < m; i++){
		betaML[i] = 0; 
		for (int k = 0; k < numCoef; k++){
			betaML[i] = betaML[i] + sample[k]/pow((k+1), 1)*cos(M_PI*(k+1)*yCoordinate[i]); 
		}
		betaML[i] = exp(betaML[i]);
	}
}


double channelFlowAdjoint::adSolver(){
	// Compute PRPW
	double* wP = new double[3*m];
	double* PR = new double[3*m];
	for (int i  = 0; i < m; i++){
		wP[i]     = xVelocity[i].value();
		wP[i+m]   = omega[i].value();
		wP[i+2*m] = k[i].value();
	}
	trace_on(0);
	for (int i = 0; i < m; i++){
		xVelocity[i] <<= wP[i];
	}
	for (int i = 0; i < m; i++){
		omega[i] <<= wP[i+m];
	}
	for (int i = 0; i < m; i++){
		k[i] <<= wP[i+2*m];
	}
	residualUpdate();
	for (int i = 0; i < 3*m; i++){
		R[i] >>= PR[i];
	}
	trace_off(0);
	double* PRPW[3*m];
	for (int i = 0; i < 3*m; i++){
		PRPW[i] = new double[3*m];
	}
	jacobian(0, 3*m, 3*m, wP, PRPW);

	//Compute PJPW
	double Py = 0.0;
	for (int i  = 0; i < m; i++){
		wP[i]     = xVelocity[i].value();
		wP[i+m]   = omega[i].value();
		wP[i+2*m] = k[i].value();
	}		
	adouble J = 0;
	trace_on(0);
	for (int i = 0; i < m; i++){
		xVelocity[i] <<= wP[i];
	}
	for (int i = 0; i < m; i++){
		omega[i] <<= wP[i+m];
	}
	for (int i = 0; i < m; i++){
		k[i] <<= wP[i+2*m];
	}
	int idx = 0;
	for (int i = 0; i < 577; i++){
		while (validationYCoordinate[i] > yCoordinate[idx+1]){
			idx++;
		}
		J = J + 1e6*pow((xVelocity[idx+1] - (xVelocity[idx+1] - xVelocity[idx])/(yCoordinate[idx+1] - yCoordinate[idx])*(yCoordinate[idx+1] - validationYCoordinate[i]))-validationVelocity[i], 2);
	}
	J >>= Py;
	trace_off(0);
	double* PJPW = new double[3*m];
	gradient(0, 3*m, wP, PJPW);

	// Compute PRPBeta
	double* betaP = new double[3*m];
	for (int i  = 0; i < m; i++){
		betaP[i]     = betaML[i].value();
	}
	trace_on(0);
	for (int i = 0; i < m; i++){
		betaML[i] <<= betaP[i];
	}
	residualUpdate();
	for (int i = 0; i < 3*m; i++){
		R[i] >>= PR[i];
	}
	trace_off(0);
	double* PRPBeta[3*m];
	for (int i = 0; i < 3*m; i++){
		PRPBeta[i] = new double[m];
	}
	jacobian(0, 3*m, m, betaP, PRPBeta);

	//Set up for petsc adjoint solver
	for (int i = 0; i < 3*m; i++){
		for (int j = 0; j < 3*m; j++){
			MatSetValue(PRPWMat, i, j, PRPW[i][j], INSERT_VALUES);
		}
		for (int j = 0; j < m; j++){
			MatSetValue(PRPBetaMat, i, j, PRPBeta[i][j], INSERT_VALUES);
		}
		VecSetValue(PJPWVec, i, PJPW[i], INSERT_VALUES);
	}
	for (int i = 0; i < m; i++){
		VecSetValue(PJPBetaVec, i, 1*(betaML[i].value()-1), INSERT_VALUES);
	}

	MatAssemblyBegin(PRPWMat, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(PRPWMat, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(PRPBetaMat, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(PRPBetaMat, MAT_FINAL_ASSEMBLY);
	VecAssemblyBegin(PJPWVec);
	VecAssemblyEnd(PJPWVec);
	VecAssemblyBegin(PJPBetaVec);
	VecAssemblyEnd(PJPBetaVec);

	// MatView(PRPWMat, PETSC_VIEWER_STDOUT_WORLD);
	// VecView(PJPWVec, PETSC_VIEWER_STDOUT_WORLD);
	// VecView(PJPBetaVec, PETSC_VIEWER_STDOUT_WORLD);

	//Solve for adjoints
	KSPSetOperators(ksp,PRPWMat,PRPWMat);
	KSPGetPC(ksp,&pc);
	PCSetType(pc,PCJACOBI);
	KSPSetTolerances(ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
	KSPSetFromOptions(ksp);
	KSPGMRESSetRestart(ksp, restart);
	KSPSolveTranspose(ksp,PJPWVec,adjoints);

	// VecView(adjoints, PETSC_VIEWER_STDOUT_WORLD);

	//Compute gradient;
	MatMultTranspose(PRPBetaMat, adjoints, adjointTerm);
	VecAXPY(PJPBetaVec, -1, adjointTerm);

	//Compute low-rank hessian approximation
	Mat yMat;
	MatCreateSeqDense(PETSC_COMM_WORLD, m, 12, NULL, &yMat);
	for (int i = 0; i < 3*m; i++){
		for (int j = 0; j < 3*m; j++){
			if (i == j && i < m){
				MatSetValue(obsMat, i, j, 1e6, INSERT_VALUES);
			} else {
				MatSetValue(obsMat, i, j, 0, INSERT_VALUES);
			}
		}
	}
	MatAssemblyBegin(obsMat, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(obsMat, MAT_FINAL_ASSEMBLY);

	//Compute with randomize algorithm
	Mat randomOmega;
	MatCreate(PETSC_COMM_WORLD, &randomOmega);
	MatSetSizes(randomOmega, PETSC_DECIDE, PETSC_DECIDE, m, 12);
	MatSetFromOptions(randomOmega);
	MatSetUp(randomOmega);
	for (int i = 0; i < m; i++){
		for (int j = 0; j < 12; j++){
			MatSetValue(randomOmega, i, j, distribution(generator), INSERT_VALUES);
		}
	}
	MatAssemblyBegin(randomOmega, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(randomOmega, MAT_FINAL_ASSEMBLY);

	for (int i = 0; i < 12; i++){
		MatGetColumnVector(randomOmega, inputSpace, i);
		MatMult(PRPBetaMat, inputSpace, workSpace1);
		KSPSolve(ksp, workSpace1, workSpace2);
		MatMult(obsMat, workSpace2, workSpace1);
		KSPSolveTranspose(ksp, workSpace1, workSpace2);
		MatMultTranspose(PRPBetaMat, workSpace2, inputSpace);
		PetscScalar *outputSpace;
		VecGetArray(inputSpace, &outputSpace);
		for (int j = 0; j < m; j++){
			MatSetValue(yMat, j, i, outputSpace[j], INSERT_VALUES);
		}
	}
	MatAssemblyBegin(yMat, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(yMat, MAT_FINAL_ASSEMBLY);

	BV bv;
	Mat Q;
	BVCreateFromMat(yMat, &bv);
	BVSetFromOptions(bv);
	BVOrthogonalize(bv, NULL);
	BVCreateMat(bv, &Q);

	EPS eps;
	Mat T, S, lambda, U, UTranspose, ReducedHessian;
	Vec workSpace3, xr, xi;
	PetscScalar kr, ki;
	MatCreate(PETSC_COMM_WORLD, &T);
	MatCreate(PETSC_COMM_WORLD, &S);
	MatCreate(PETSC_COMM_WORLD, &U);
	MatCreate(PETSC_COMM_WORLD, &lambda);
	VecCreate(PETSC_COMM_WORLD, &workSpace3);
	MatSetSizes(T, PETSC_DECIDE, PETSC_DECIDE, 12, 12);
	MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, 12, 12);
	MatSetSizes(U, PETSC_DECIDE, PETSC_DECIDE, 12, 12);
	MatSetSizes(lambda, PETSC_DECIDE, PETSC_DECIDE, 12, 12);
	VecSetSizes(workSpace3, PETSC_DECIDE, 12);
	MatSetFromOptions(T);
	MatSetFromOptions(S);
	MatSetFromOptions(U);
	MatSetFromOptions(lambda);
	VecSetFromOptions(workSpace3);
	MatSetUp(T);
	MatSetUp(S);
	MatSetUp(lambda);
	MatSetUp(U);
	VecSetUp(workSpace3);

	for (int i = 0; i < 12; i++){
		MatGetColumnVector(Q, inputSpace, i);
		MatMult(PRPBetaMat, inputSpace, workSpace1);
		KSPSolve(ksp, workSpace1, workSpace2);
		MatMult(obsMat, workSpace2, workSpace1);
		KSPSolveTranspose(ksp, workSpace1, workSpace2);
		MatMultTranspose(PRPBetaMat, workSpace2, inputSpace);
		MatMultTranspose(Q, inputSpace, workSpace3);
		PetscScalar *outputSpace;
		VecGetArray(workSpace3, &outputSpace);
		for (int j = 0; j < 12; j++){
			MatSetValue(T, j, i, outputSpace[j], INSERT_VALUES);
		}
	}
	MatAssemblyBegin(T, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(T, MAT_FINAL_ASSEMBLY);
	MatCreateVecs(T, NULL, &xr);
	MatCreateVecs(T, NULL, &xi);

	EPSCreate(PETSC_COMM_WORLD, &eps);
	EPSSetOperators(eps, T, NULL);
	EPSSetFromOptions(eps);
	EPSSolve(eps);

	PetscScalar *eigenValues;
	for (int i = 0; i < 12; i++){
		EPSGetEigenpair(eps, i, &kr, &ki, xr, xi);
		VecGetArray(xr, &eigenValues);
		for (int j = 0; j < 12; j++){
			if (i == j){
				MatSetValue(lambda, j, i, kr, INSERT_VALUES);
			} else {
				MatSetValue(lambda, j, i, 0, INSERT_VALUES);
			}
			MatSetValue(S, j, i, eigenValues[j], INSERT_VALUES);
		}
	}

	MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(lambda, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(lambda, MAT_FINAL_ASSEMBLY);

	MatMatMult(Q, S, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &U);
	MatTranspose(U, MAT_INITIAL_MATRIX, &UTranspose);
	MatPtAP(lambda, UTranspose, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &ReducedHessian);

	// Compute with Krylov Subspace Method
	// PetscInt n = m;
	// Mat Action; 
	// EPS eps2;
	// MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, m, &n, &Action);
	// MatShellSetOperation(Action, MATOP_MULT, (void(*)(void))MatMult_Shell);  //Problems with TypeCasting
	// MatShellSetOperation(Action, MATOP_MULT_TRANSPOSE, (void(*)(void))MatMult_Shell);  //Problems with TypeCasting
	// MatShellSetOperation(Action, MATOP_GET_DIAGONAL, (void(*)(void))MatGetDiagonal_Shell);  //Problems with TypeCasting

	// EPSCreate(PETSC_COMM_WORLD, &eps2);
	// EPSSetOperators(eps2, Action, NULL);
	// EPSSetProblemType(eps, EPS_HEP);
	// EPSSetDimensions(eps2, 10, PETSC_DEFAULT, PETSC_DEFAULT);
	// EPSSetFromOptions(eps2);
	// EPSSolve(eps);
	

	VecView(PJPBetaVec, PETSC_VIEWER_STDOUT_WORLD);
	MatView(ReducedHessian, PETSC_VIEWER_STDOUT_WORLD);

	// PetscScalar *gradient;
	// VecGetArray(PJPBetaVec, &gradient);
	// for (int i = 0; i < m; i++){
	// 	std::cout << gradient[i] << " ";
	// }
	// std::cout << std::endl;

	// PetscScalar *ReducedHessianColumn;
	// for (int i = 0; i < m; i++){
	// 	MatGetColumnVector(ReducedHessian, inputSpace, i);
	// 	VecGetArray(inputSpace, &ReducedHessianColumn);
	// 	for (int j = 0; j < m; j++){
	// 		std::cout << ReducedHessianColumn[j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// for (int i = 0; i < numCoef; i++){
	// 	gradientVector[i] = g[i];
	// 	for (int j = 0; j < i+1; j++){
	// 		hessianMatrix[i*numCoef+j] = h[i][j];			
	// 		hessianMatrix[j*numCoef+i] = h[i][j];
	// 	}
	// }

	return 0;
}

// PetscErrorCode channelFlowAdjoint::MatMult_Shell(Mat Action, Vec input, Vec output){
// 	PetscFunctionBeginUser;
// 	MatMult(PRPBetaMat, input, workSpace1);
// 	KSPSolve(ksp, workSpace1, workSpace2);
// 	MatMult(obsMat, workSpace2, workSpace1);
// 	KSPSolveTranspose(ksp, workSpace1, workSpace2);
// 	MatMultTranspose(PRPBetaMat, workSpace2, output);

// 	PetscFunctionReturn(0);
// };

// PetscErrorCode channelFlowAdjoint::MatGetDiagonal_Shell(Mat Action, Vec diag){
// 	PetscFunctionBeginUser;
// 	PetscScalar *outputSpace;
// 	for (int i = 0; i < m; i++){
// 		VecSet(inputSpace, 0);
// 		VecSetValue(inputSpace, i, 1, INSERT_VALUES);
// 		VecAssemblyBegin(inputSpace);
// 		VecAssemblyEnd(inputSpace);
// 		MatMult(PRPBetaMat, inputSpace, workSpace1);
// 		KSPSolve(ksp, workSpace1, workSpace2);
// 		MatMult(obsMat, workSpace2, workSpace1);
// 		KSPSolveTranspose(ksp, workSpace1, workSpace2);
// 		MatMultTranspose(PRPBetaMat, workSpace2, inputSpace);
// 		VecGetArray(inputSpace, &outputSpace);
// 		VecSetValue(diag, i, outputSpace[i], INSERT_VALUES);
// 	}
// 	VecAssemblyBegin(diag);
// 	VecAssemblyEnd(diag);

// 	PetscFunctionReturn(0);
// };