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

#include <linearAlgebra.h>

void KronProduct(Mat A, int an, int am, Mat B, int bn, int bm, Mat C){
	double entry;
	int idxm[1] = {0};
	int idxn[1] = {0};
	double v[1] = {0};
	double w[1] = {0};
	for (int i = 0; i < an; ++i){
		for (int j = 0; j < bn; ++j){
			for (int m = 0; m < am; ++m){
				for (int n = 0; n < bm; ++n){
					idxm[0] = i;
					idxn[0] = m;
					MatGetValues(A, 1, idxm, 1, idxn, v);
					idxm[0] = j;
					idxn[0] = n;
					MatGetValues(B, 1, idxm, 1, idxn, w);
					entry = w[0]*v[0];
					if (entry != 0){
						MatSetValue(C, i*an+j, m*am+n, entry, INSERT_VALUES);
					}
				}
			}
		}
	}
	MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);
}

void cheb(int N, Mat D, double x[]){
	double c[N+1];
	for (int i = 0; i < N+1; ++i){
		x[i] = cos(M_PI*i/N);
	}
	c[0] = 2;
	c[N] = 2*pow(-1, N);
	for (int i = 1; i< N; ++i){
		c[i] = pow(-1, i);
	}
	double eye;
	double entry;
	double sum[N+1];
	for (int i = 0; i < N+1; ++i){
		sum[i] = 0;
		for (int j = 0; j < N+1; ++j){
			if (i == j) {
				eye = 1;
			} else {
				eye = 0;
			}
			entry = c[i]*1.0/c[j]/(x[i]-x[j]+eye);
			sum[i] += entry;
			MatSetValue(D, i, j, entry, INSERT_VALUES);
		}
	}
	MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);
	for (int i = 0; i < N+1; ++i){
		MatSetValue(D, i, i, -sum[i], ADD_VALUES);
	}
	MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);
}

void clenshawCurtis(int N, double w[]){
	Mat localMatrix;
	MatCreate(PETSC_COMM_SELF, &localMatrix);
	MatSetSizes(localMatrix, PETSC_DECIDE, PETSC_DECIDE, N/2+1, N/2+1);
	MatSetFromOptions(localMatrix);
	MatSeqAIJSetPreallocation(localMatrix, N/2+1, NULL);
	MatMPIAIJSetPreallocation(localMatrix, N/2+1, NULL, N/2+1, NULL);
	double cn;
	for (int k = 0; k < N/2+1; ++k){
		for (int n = 0; n < N/2+1; ++n){
			if (n == 0 || n == N/2){
				cn = 0.5;
			} else {
				cn = 1.0;
			}
			MatSetValue(localMatrix, k, n, 2.0/N*cos(n*k*M_PI/N*2.0)*cn, INSERT_VALUES);
		}
	}
	MatAssemblyBegin(localMatrix, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(localMatrix, MAT_FINAL_ASSEMBLY);

	Vec localVector;
	VecCreate(PETSC_COMM_SELF, &localVector);
	VecSetSizes(localVector, PETSC_DECIDE, N/2+1);
	VecSetFromOptions(localVector);

	VecSetValue(localVector, 0, 1.0, INSERT_VALUES);
	VecSetValue(localVector, N/2, 1.0/(1.0-pow(N, 2)), INSERT_VALUES);
	for (int i = 1; i < N/2; ++i){
		VecSetValue(localVector, i, 2.0/(1-pow((2*i),2)), INSERT_VALUES);
	}
	VecAssemblyBegin(localVector);
	VecAssemblyEnd(localVector);

	Vec localVector2;
	VecDuplicate(localVector, &localVector2);
	MatMultTranspose(localMatrix, localVector, localVector2);
	double *localArray;
	VecGetArray(localVector2, &localArray);
	for (int i = 0; i < N/2; ++i){
		w[i] = localArray[i];
	}
	w[N/2] = 2*localArray[N/2];
	for (int i = 0; i < N/2; ++i){
		w[N/2+1+i] = w[N/2-1-i];
	}
	MatDestroy(&localMatrix);
	VecDestroy(&localVector);
	VecDestroy(&localVector2);
}

PetscErrorCode MyKSPMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *dummy)
{
	PetscPrintf(PETSC_COMM_SELF, "iteration %D KSP Residual norm %14.12e \n", n, rnorm);
	return 0;
}

class stokesChebSolver {
private:
	std::unique_ptr<double[]> x;
	std::unique_ptr<double[]> u1;
	std::unique_ptr<double[]> u2;
	
	Mat     D;
	Mat     DD;
	Mat     D0;
	Mat     DD0;
	Mat     Dtilda;
	Mat     I;
	Mat     I0;
	Mat     A;
	Mat 	systemNest;
	Mat 	systemMatrix;
	Mat     A00;
	Mat     A11;
	Mat 	A02;
	Mat 	A12;
	Mat 	A20;
	Mat 	A21;

	Vec     rhs;
	Vec     states;
	Vec     u1Full;
	Vec     u2Full;

	KSP     ksp;
	PC      pc;

	PetscInt *indices;
	PetscInt *indices1;
	PetscInt *indices2;
	PetscInt *indices3;

	IS      isrow;
	IS      iscol;
	IS      A00IS;
	IS      A11IS;
	IS      A22IS;

public:
	int     numSpectrals;
	double  m;

	stokesChebSolver(int numSpectrals_, double m_);
	~stokesChebSolver(){
		MatDestroy(&D);
		MatDestroy(&DD);
		MatDestroy(&D0);
		MatDestroy(&DD0);
		MatDestroy(&Dtilda);
		MatDestroy(&A);
		MatDestroy(&I0);
		MatDestroy(&I);
		MatDestroy(&A00);
		MatDestroy(&A11);
		MatDestroy(&A02);
		MatDestroy(&A12);		
		MatDestroy(&A20);
		MatDestroy(&A21);
		MatDestroy(&systemNest);
		MatDestroy(&systemMatrix);

		VecDestroy(&rhs);
		VecDestroy(&states);
		VecDestroy(&u1Full);
		VecDestroy(&u2Full);

		ISDestroy(&isrow);
		ISDestroy(&iscol);
		ISDestroy(&A00IS);
		ISDestroy(&A11IS);
		ISDestroy(&A22IS);

		KSPDestroy(&ksp);
	};

	void updateRHS(double c);
	void solve();
	void output(double& obs, double& qoi);
};

stokesChebSolver::stokesChebSolver(int numSpectrals_, double m_) : numSpectrals(numSpectrals_) , m(m_) {
	//Initialize chebyshev differentiation matrix
	int N = numSpectrals_;
	MatCreate(PETSC_COMM_SELF, &D);
	MatSetSizes(D, PETSC_DECIDE, PETSC_DECIDE, N+1, N+1);
	MatSetFromOptions(D);
	MatSeqAIJSetPreallocation(D, N+1, NULL);
	MatMPIAIJSetPreallocation(D, N+1, NULL, N+1, NULL);
	x = std::make_unique<double[]>(N+1);

	cheb(N, D, x.get());
	for (int i = 0; i < N+1; ++i){
		x[i] = -0.5*x[i] + 0.5;
	}

	MatScale(D, 2);
	MatMatMult(D, D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DD);

	int size = N-1;
	PetscMalloc1(size, &indices);
	for (int i = 1; i < size+1; i++){
		indices[i-1] = i;
	}
	ISCreateGeneral(PETSC_COMM_SELF, size, indices, PETSC_COPY_VALUES, &isrow);
	ISCreateGeneral(PETSC_COMM_SELF, size, indices, PETSC_COPY_VALUES, &iscol);

	MatCreateSubMatrix(D, isrow, iscol, MAT_INITIAL_MATRIX, &D0);
	MatCreateSubMatrix(DD, isrow, iscol, MAT_INITIAL_MATRIX, &DD0);

	MatCreate(PETSC_COMM_SELF, &I);
	MatSetSizes(I, PETSC_DECIDE, PETSC_DECIDE, N+1, N+1);
	MatSetFromOptions(I);
	MatSeqAIJSetPreallocation(I, 2, NULL);
	MatMPIAIJSetPreallocation(I, 2, NULL, 2, NULL);
	for (int i = 0; i < N+1; ++i){
		MatSetValue(I, i, i, 1, INSERT_VALUES);
	}

	MatCreate(PETSC_COMM_SELF, &I0);
	MatSetSizes(I0, PETSC_DECIDE, PETSC_DECIDE, N-1, N-1);
	MatSetFromOptions(I0);
	MatSeqAIJSetPreallocation(I0, 2, NULL);
	MatMPIAIJSetPreallocation(I0, 2, NULL, 2, NULL);
	for (int i = 0; i < N-1; ++i){
		MatSetValue(I0, i, i, 1, INSERT_VALUES);
	}

	MatAssemblyBegin(I, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(I, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(I0, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(I0, MAT_FINAL_ASSEMBLY);

	//Create system matrix
	MatCreate(PETSC_COMM_SELF, &A);
	MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, pow(N-1, 2)*3, pow(N-1, 2)*3);
	MatSetFromOptions(A);
	MatSeqAIJSetPreallocation(A, 3*N, NULL);
	MatMPIAIJSetPreallocation(A, 3*N, NULL, 3*N, NULL);
	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

	Mat localWorkSpace1;
	MatCreate(PETSC_COMM_SELF, &localWorkSpace1);
	MatSetSizes(localWorkSpace1, PETSC_DECIDE, PETSC_DECIDE, pow(N-1, 2), pow(N-1, 2));
	MatSetFromOptions(localWorkSpace1);
	MatSeqAIJSetPreallocation(localWorkSpace1, N, NULL);
	MatMPIAIJSetPreallocation(localWorkSpace1, N, NULL, N, NULL);	

	Mat localWorkSpace2;
	MatCreate(PETSC_COMM_SELF, &localWorkSpace2);
	MatSetSizes(localWorkSpace2, PETSC_DECIDE, PETSC_DECIDE, pow(N-1, 2), pow(N-1, 2));
	MatSetFromOptions(localWorkSpace2);
	MatSeqAIJSetPreallocation(localWorkSpace2, N, NULL);
	MatMPIAIJSetPreallocation(localWorkSpace2, N, NULL, N, NULL);	

	KronProduct(DD0, N-1, N-1, I0, N-1, N-1, localWorkSpace1);
	KronProduct(I0, N-1, N-1, DD0, N-1, N-1, localWorkSpace2);

	int final_size = pow(N-1, 2);
	PetscMalloc1(final_size, &indices1);
	PetscMalloc1(final_size, &indices2);
	PetscMalloc1(final_size, &indices3);
	for (int i = 0; i < final_size; i++){
		indices1[i] = i;
		indices2[i] = i+pow(N-1, 2);
		indices3[i] = i+2*pow(N-1, 2);
	}
	ISCreateGeneral(PETSC_COMM_SELF, final_size, indices1, PETSC_COPY_VALUES, &A00IS);
	ISCreateGeneral(PETSC_COMM_SELF, final_size, indices2, PETSC_COPY_VALUES, &A11IS);
	ISCreateGeneral(PETSC_COMM_SELF, final_size, indices3, PETSC_COPY_VALUES, &A22IS);	

	MatCreateSubMatrix(A, A00IS, A00IS, MAT_INITIAL_MATRIX, &A00);
	MatCreateSubMatrix(A, A11IS, A11IS, MAT_INITIAL_MATRIX, &A11);

	MatAXPY(A00, 1, localWorkSpace1, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(A00, 1, localWorkSpace2, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(A11, 1, localWorkSpace1, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(A11, 1, localWorkSpace2, DIFFERENT_NONZERO_PATTERN);

	double entry;

	MatCreate(PETSC_COMM_SELF, &Dtilda);
	MatSetSizes(Dtilda, PETSC_DECIDE, PETSC_DECIDE, N-1, N-1);
	MatSetFromOptions(Dtilda);
	MatSeqAIJSetPreallocation(Dtilda, N-1, NULL);
	MatMPIAIJSetPreallocation(Dtilda, N-1, NULL, N-1, NULL);

	for(int i = 1; i < N; ++i){
		for (int j = 1; j < N; ++j){
			if (i == j){
				entry = 3.0*cos(M_PI*i/N)/2.0/(pow(sin(M_PI*i/N), 2.0));
			} else {
				entry = -1.0/2.0*pow(sin(M_PI*j/N), 2.0)*pow(-1.0, i+j)/(pow(sin(M_PI*i/N), 2.0)*sin(M_PI*(i+j)/2.0/N)*sin(M_PI*(i-j)/2.0/N));
			}
			MatSetValue(Dtilda, i-1, j-1, entry, INSERT_VALUES);
		}
	}
	MatAssemblyBegin(Dtilda, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(Dtilda, MAT_FINAL_ASSEMBLY);
	MatScale(Dtilda, 2);

	MatCreate(PETSC_COMM_SELF, &A02);
	MatSetSizes(A02, PETSC_DECIDE, PETSC_DECIDE, pow(N-1, 2), pow(N-1, 2));
	MatSetFromOptions(A02);
	MatSeqAIJSetPreallocation(A02, N-1, NULL);
	MatMPIAIJSetPreallocation(A02, N-1, NULL, N-1, NULL);	

	MatCreate(PETSC_COMM_SELF, &A12);
	MatSetSizes(A12, PETSC_DECIDE, PETSC_DECIDE, pow(N-1, 2), pow(N-1, 2));
	MatSetFromOptions(A12);
	MatSeqAIJSetPreallocation(A12, N-1, NULL);
	MatMPIAIJSetPreallocation(A12, N-1, NULL, N-1, NULL);	

	MatCreate(PETSC_COMM_SELF, &A20);
	MatSetSizes(A20, PETSC_DECIDE, PETSC_DECIDE, pow(N-1, 2), pow(N-1, 2));
	MatSetFromOptions(A20);
	MatSeqAIJSetPreallocation(A20, N-1, NULL);
	MatMPIAIJSetPreallocation(A20, N-1, NULL, N-1, NULL);	

	MatCreate(PETSC_COMM_SELF, &A21);
	MatSetSizes(A21, PETSC_DECIDE, PETSC_DECIDE, pow(N-1, 2), pow(N-1, 2));
	MatSetFromOptions(A21);
	MatSeqAIJSetPreallocation(A21, N-1, NULL);
	MatMPIAIJSetPreallocation(A21, N-1, NULL, N-1, NULL);	

	KronProduct(Dtilda, N-1,N-1,I0, N-1,N-1,A12);
	KronProduct(I0, N-1,N-1,Dtilda, N-1,N-1,A02);	
	KronProduct(D0, N-1,N-1,I0, N-1,N-1,A21);
	KronProduct(I0, N-1,N-1,D0, N-1,N-1,A20);	

	Mat sublist[9] = {A00, NULL, A02, NULL, A11, A12, A20, A21, NULL};
	MatCreateNest(PETSC_COMM_SELF, 3, NULL, 3, NULL, sublist, &systemNest);
	MatConvert(systemNest, MATSEQAIJ, MAT_INITIAL_MATRIX, &systemMatrix);

	double f[N-1];
	double b[(int) pow(N-1, 2)*3] = {0};

	for (int i = 0; i < N-1; ++i){
		f[i] = sin(2*M_PI*x[i+1]);
	}
	for (int i = 0; i < N-1; ++i){
		for (int j = 0; j < N-1; ++j){
			b[i*(N-1)+j] = f[i]*f[j];
			b[(int) pow(N-1, 2)+i*(N-1)+j] = f[i]*f[j];
		}
	}
	VecCreate(PETSC_COMM_SELF, &rhs);
	VecSetSizes(rhs, PETSC_DECIDE, 3*pow(N-1, 2));
	VecSetFromOptions(rhs);
	for (int i = 0; i < pow(N-1, 2)*3; ++i){
		VecSetValue(rhs, i, m_*100.0*b[i], INSERT_VALUES);
	}
	VecAssemblyBegin(rhs);
	VecAssemblyEnd(rhs);

	VecDuplicate(rhs, &states);

	KSPCreate(PETSC_COMM_SELF, &ksp);
	KSPSetType(ksp, KSPFGMRES);
	KSPSetOperators(ksp, systemMatrix, systemMatrix);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCFIELDSPLIT);
	PCFieldSplitSetDetectSaddlePoint(pc, PETSC_TRUE);
	PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELF, NULL);
	PCSetUp(pc);


	KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPMonitorSet(ksp, MyKSPMonitor, NULL, 0);
	KSPSetUp(ksp);
	// KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);

	u1 = std::make_unique<double[]>(pow(N-1, 2));
	u2 = std::make_unique<double[]>(pow(N-1, 2));

	MatDestroy(&localWorkSpace1);
	MatDestroy(&localWorkSpace2);

	PetscFree(indices);
	PetscFree(indices1);
	PetscFree(indices2);
	PetscFree(indices3);
};

void stokesChebSolver::updateRHS(double c){
	int N = numSpectrals;
	m = c;

	double f[N-1];
	double b[(int) pow(N-1, 2)*3] = {0};

	for (int i = 0; i < N-1; ++i){
		f[i] = sin(2*M_PI*x[i+1]);
	}
	for (int i = 0; i < N-1; ++i){
		for (int j = 0; j < N-1; ++j){
			b[i*(N-1)+j] = f[i]*f[j];
			b[(int) pow(N-1, 2)+i*(N-1)+j] = f[i]*f[j];
		}
	}
	for (int i = 0; i < pow(N-1, 2)*3; ++i){
		VecSetValue(rhs, i, c*100.0*b[i], INSERT_VALUES);
	}
	VecAssemblyBegin(rhs);
	VecAssemblyEnd(rhs);
}

void stokesChebSolver::solve(){
	double N = numSpectrals;
	double *states_array;
	KSPSolve(ksp, rhs, states);
	VecGetArray(states, &states_array);
	std::copy(states_array, states_array+ (int) pow(N-1, 2), u1.get());
	std::copy(states_array+(int)pow(N-1, 2), states_array+2*(int)pow(N-1, 2), u2.get());	
}

void stokesChebSolver::output(double& obs, double& qoi){
	int N = numSpectrals;

	Mat localWorkSpace1;
	MatCreate(PETSC_COMM_SELF, &localWorkSpace1);
	MatSetSizes(localWorkSpace1, PETSC_DECIDE, PETSC_DECIDE, pow(N+1, 2), pow(N+1, 2));
	MatSetFromOptions(localWorkSpace1);
	MatSeqAIJSetPreallocation(localWorkSpace1, 2*N, NULL);
	MatMPIAIJSetPreallocation(localWorkSpace1, 2*N, NULL, 2*N, NULL);	

	Mat localWorkSpace2;
	MatCreate(PETSC_COMM_SELF, &localWorkSpace2);
	MatSetSizes(localWorkSpace2, PETSC_DECIDE, PETSC_DECIDE, pow(N+1, 2), pow(N+1, 2));
	MatSetFromOptions(localWorkSpace2);
	MatSeqAIJSetPreallocation(localWorkSpace2, 2*N, NULL);
	MatMPIAIJSetPreallocation(localWorkSpace2, 2*N, NULL, 2*N, NULL);	

	KronProduct(D, N+1, N+1, I, N+1, N+1, localWorkSpace1);
	KronProduct(I, N+1, N+1, D, N+1, N+1, localWorkSpace2);

	VecCreate(PETSC_COMM_SELF ,&u1Full);
	VecCreate(PETSC_COMM_SELF, &u2Full);
	VecSetSizes(u1Full, PETSC_DECIDE, pow(N+1, 2));
	VecSetSizes(u2Full, PETSC_DECIDE, pow(N+1, 2));
	VecSetFromOptions(u1Full);
	VecSetFromOptions(u2Full);
	int counter = 0;
	for (int i = 1; i < N; ++i){
		for (int j = 1; j < N; ++j){
			VecSetValue(u1Full, i*(N+1)+j, u1[counter], INSERT_VALUES);
			VecSetValue(u2Full, i*(N+1)+j, u2[counter], INSERT_VALUES);
			counter++;
		}
	}
	VecAssemblyBegin(u1Full);
	VecAssemblyEnd(u1Full);
	VecAssemblyBegin(u2Full);
	VecAssemblyEnd(u2Full);

	Vec ux_y;
	Vec uy_x;
	VecDuplicate(u1Full, &ux_y);
	VecDuplicate(u2Full, &uy_x);
	MatMult(localWorkSpace1, u1Full, ux_y);
	MatMult(localWorkSpace2, u2Full, uy_x);

	double *local_ux_y;
	double *local_uy_x;
	VecGetArray(ux_y, &local_ux_y);
	VecGetArray(uy_x, &local_uy_x);

	double qoi_ux[(int)pow(N+1, 2)];
	double qoi_uy[(int)pow(N+1, 2)];
	double obs_ux[(int)pow(N+1, 2)];
	double obs_uy[(int)pow(N+1, 2)];
	for (int i = 0; i < N+1; ++i){
		for (int j = 0; j < N+1; ++j){
			obs_ux[i*(N+1)+j] = local_ux_y[i*(N+1)+j]*pow(x[i], 1.5);
			qoi_ux[i*(N+1)+j] = local_ux_y[i*(N+1)+j]*pow(x[i], 0.5);	
		}
	}
	for (int i = 0; i < N+1; ++i){
		for (int j = 0; j < N+1; ++j){
			obs_uy[j*(N+1)+i] = local_uy_x[j*(N+1)+i]*pow(x[i], 1.5);
			qoi_uy[j*(N+1)+i] = local_uy_x[j*(N+1)+i]*pow(x[i], 0.5);			
		}
	}

	double w[N+1];
	clenshawCurtis(N, w);

	obs = 0; 
	qoi = 0;
	for (int i = 0; i < N+1; ++i){
		for (int j = 0; j < N+1; ++j){
			obs = obs + 100.0*w[i]*w[j]/4.0*(obs_ux[i*(N+1)+j]+obs_uy[i*(N+1)+j]);		
			qoi = qoi + 100.0*w[i]*w[j]/4.0*(qoi_ux[i*(N+1)+j]+qoi_uy[i*(N+1)+j]);		
		}
	}
	std::cout << obs << " " << qoi << std::endl;
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


	// std::vector<std::string> name;
	// std::vector<std::string> value;
	// std::vector<double> rand_coef;
	// read_config(name, value);
	// int num_term = std::stoi(value[0]);
	// for (int i = 0; i < num_term; ++i){
	// 	rand_coef.push_back(std::stod(value[i+1]));
	// }
	// int levels = std::stoi(value[num_term+1]);
	// int a = std::stoi(value[num_term+2]);
	// double pCNstep = std::stod(value[num_term+3]);
	// int task = std::stoi(value[num_term+4]);
	// int plainMCMC_sample_number = std::stoi(value[num_term+5]);
	// int obsNumPerRow = std::stoi(value[num_term+6]);
	// double noiseVariance = std::stod(value[num_term+7]);

	// if (rank == 0){
	// 	std::cout << "configuration: " << std::endl; 
	// 	std::cout << "num_term: " << num_term << " coefs: ";
	// 	for (int i = 0; i < num_term; ++i){
	// 		std::cout << rand_coef[i] << " ";
	// 	}
	// 	std::cout << std::endl;
	// 	std::cout << "levels: "           << levels       <<  std::endl;
	// 	std::cout << "a: "                << a            <<  std::endl;
	// 	std::cout << "pCNstep: "          << pCNstep          <<  std::endl;
	// 	std::cout << "task: "             << task         <<  std::endl;
	// 	std::cout << "plainMCMC samples:" << plainMCMC_sample_number << std::endl;
	// 	std::cout << "obsNumPerRow: "     << obsNumPerRow <<  std::endl;
	// 	std::cout << "noiseVariance: "    << noiseVariance << std::endl; 
	// }

	// if (task == 2){
	// 	double output;
	// 	// MLMCMC_Bi_Uniform<pCN<stokesSolver>, stokesSolver> MLMCMCSolver(levels, 1, rank, a, noiseVariance, 1.0);
	// 	// output = MLMCMCSolver.mlmcmcRun();

	// 	MLMCMC_Bi<pCN<stokesSolver>, stokesSolver> MLMCMCSolver(levels, 1, rank, a, noiseVariance, 1.0);
	// 	output = MLMCMCSolver.mlmcmcRun();
	// 	std::cout << output << std::endl;

	// 	std::string outputfile = "output_";
	// 	outputfile.append(std::to_string(rank));

	// 	std::ofstream myfile;
	// 	myfile.open(outputfile);
	// 	for (int i = 0; i<num_term; ++i){
	// 		myfile << output << " ";
	// 	}
	// 	myfile << std::endl;
	// 	myfile.close();

	// 	MPI_Barrier(MPI_COMM_WORLD);
	// 	if (rank == 0){
	// 		double buffer;
	// 		std::string finaloutput = "finalOutput";
	// 		std::ofstream outputfile;
	// 		outputfile.open(finaloutput);
	// 		for (int i = 0; i < size; i++){
	// 			std::string finalinput = "output_";
	// 			finalinput.append(std::to_string(i));
	// 			std::ifstream inputfile;
	// 			inputfile.open(finalinput, std::ios_base::in);
	// 			for(int i = 0; i < num_term; ++i){
	// 				inputfile >> buffer;
	// 				outputfile << buffer << " ";
	// 			}
	// 			outputfile << std::endl;
	// 			inputfile.close();
	// 		}
	// 		outputfile.close();
	// 	}	
	// }

	double quadx[128]; //= {-0.9894, -0.9446, -0.8656, -0.7554, -0.6179, -0.4580, -0.2816, -0.0950,   0.0950,    0.2816,    0.4580,    0.6179,   0.7554,  0.8656,    0.9446,    0.9894};
	double quadw[128]; //= {0.0272, 0.0623, 0.0952, 0.1246, 0.1496, 0.1692, 0.1826, 0.1895, 0.1895,    0.1826,    0.1692,    0.1496,    0.1246, 0.0952,    0.0623,    0.0272};
	
	double obs;
	double qoi;

	double localNum=0;
	double localDen=0;

	//gauleg(128, quadx, quadw);  //gaussian legendre
	gauher2(128, quadx, quadw);     //gaussian hermite

	stokesChebSolver *refSolver;
	refSolver = new stokesChebSolver(200, 0.5152);

	refSolver->updateRHS(sqrt(2.0)*quadx[rank]);
	refSolver->solve();
	refSolver->output(obs, qoi);

	std::cout << "rank: " << rank << "x: " << quadx[rank] << " obs: " << obs << " qoi: " << qoi << std::endl;
	
	localNum += qoi*quadw[rank]*exp(-10.0*pow(obs+0.994214, 2)/2.0)+(-qoi)*quadw[rank]*exp(-10.0*pow(-obs+0.994214, 2)/2.0);
	localDen += quadw[rank]*exp(-10.0*pow(obs+0.994214, 2)/2.0)+quadw[rank]*exp(-10.0*pow(-obs+0.994214, 2)/2.0);

	double globalNum, globalDen;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&localNum, &globalNum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&localDen, &globalDen, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0){
		std::cout.precision(10);
		std::cout << globalNum << " " << globalDen << " " << globalNum/globalDen << std::endl;
	}

	delete refSolver;

	PetscFinalize();
	
	return 0;
}
