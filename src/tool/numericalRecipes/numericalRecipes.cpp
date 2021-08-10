#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>

#include "numericalRecipes.h"

void gauleg(int n, double x[], double w[]){
	double m = (n+1.0)/2.0;
	double xm = 0.0;
	double xl = 1.0;
	double z, z1, p1, p2, p3, pp;
	for (int i = 1; i < m+1; ++i){
		z = cos(M_PI*(i-0.25)/(n+0.5));
		while (1){
			p1 = 1.0;
			p2 = 0.0;
			for (int j = 1; j < n+1; ++j){
				p3 = p2;
				p2 = p1;
				p1 = ((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
			}
			pp = n*(z*p1-p2)/(z*z-1.0);
			z1 = z;
			z = z1-p1/pp;
			if (abs(z-z1) < 1e-10){
				break;
			}
		} 
		x[i-1] = xm-xl*z;
		x[n-i] = xm+xl*z;
		w[i-1] = 2.0*xl/((1.0-z*z)*pp*pp);
		w[n-i] = w[i-1];
	}
};

void gauher(int n, double x[], double w[]){
	double m = (n+1.0)/2.0;
	double z, z1, p1, p2, p3, pp;
	for (int i = 1; i < m+1; ++i){
		if (i == 1){
			z = sqrt(2.0*n + 1.0)-1.85575*pow(2.0*n+1.0, -1.0/6.0);
		} else if (i == 2){
			z = z - 1.14*pow(n, 0.426)/z;
		} else if (i == 3){
			z = 1.86*z - 0.86*x[0];
		} else if (i == 4){
			z = 1.91*z-0.91*x[1];
		} else {
			z = 2.0*z-x[i-3];
		}
		while (1){
			p1=1.0/pow(M_PI, 1.0/4.0);
			p2=0.0;
			for (int j = 1; j < n+1; ++j){
				p3=p2;
				p2=p1;
				p1=sqrt(2.0/j)*z*p2-sqrt((double)(j-1)/(double)j)*p3;
			}
			pp=sqrt(2.0*n)*p2;
			z1 = z;
			z = z1-p1/pp;
			if (abs(z-z1) < 1e-14) {
				break;
			}
		}
		x[i-1] = z;
		x[n-i] = -z;
		w[i-1] = 2.0/(pp*pp);
		w[n-i] = w[i-1];
	}
};

// void arth(double first, double increment, double arth_d[], double n){
// 	arth_d[0] = first;
// 	if (n <= 16){
// 		for (int i = 1; i < n; ++i){
// 			arth_d[i] = arth_d[i-1]+increment;
// 		}
// 	} else {
// 		for (int i = 1; i < 8; ++i){
// 			arth_d[i] = arth_d[i-1]+increment;
// 		}
// 		double temp=increment*8.0;
// 		int k = 8;
// 		double k2;
// 		while (k < n){
// 			k2 = k+k;
// 			for (int i = k; i < min(k2, n); ++i){
// 				arth_d[k+i] = temp+arth_d[i];
// 				temp=temp+temp;
// 				k = k2
// 			}
// 		}
// 	}
// }

void gauher2(int n, double x[], double w[]){
	double C1 = 0.9084064;
	double C2 = 0.05214976;
	double C3 = 0.002579930;
	double C4 = 0.003986126;
	double m = (n+1.0)/2.0;
	double anu = 2.0*n + 1.0;
	double rhs, r3, r2, theta, z;
	double z1, p1, p2, p3, pp;
	for (int i = 1; i < m+1; ++i){
		rhs = (3+(i-1)*4)*M_PI/anu;
		r3 = pow(rhs, 1.0/3.0);
		r2 = pow(r3, 2);
		theta = r3*(C1+r2*(C2+r2*(C3+r2*C4)));
		z = sqrt(anu)*cos(theta);

		while (1){
			p1=1.0/pow(M_PI, 1.0/4.0);
			p2=0.0;
			for (int j = 1; j < n+1; ++j){
				p3=p2;
				p2=p1;
				p1=sqrt(2.0/j)*z*p2-sqrt((double)(j-1)/(double)j)*p3;
			}
			pp=sqrt(2.0*n)*p2;
			z1 = z;
			z = z1-p1/pp;
			if (abs(z-z1) < 3.0e-13) {
				break;
			}
		}
		x[i-1] = z;
		x[n-i] = -z;
		w[i-1] = 2.0/(pp*pp);
		w[n-i] = w[i-1];
	}
};


void SPDRegularization(double SPDMatrix[], int dimension, double criteria){
	//declaration of variables
	gsl_matrix *originalMatrix = gsl_matrix_alloc(dimension, dimension);
	gsl_vector *eigenValues = gsl_vector_alloc(dimension);
	gsl_matrix *diagonalizedEigenValues = gsl_matrix_alloc(dimension, dimension);
	gsl_matrix *eigenVectors = gsl_matrix_alloc(dimension, dimension);
	gsl_matrix *workSpaceMatrix = gsl_matrix_alloc(dimension, dimension);

	//initialize matrix
	// std::cout << std::endl << "Hessian Matrix" << std::endl;
	for (int i = 0; i < dimension; i++){
		for (int j = 0; j < dimension; j++){
			gsl_matrix_set(originalMatrix, i, j, SPDMatrix[i*dimension+j]);
			// std::cout << SPDMatrix[i*dimension+j] << ", ";
		}
		// std::cout << std::endl;
	}

	//eigen decomposition
	gsl_eigen_symmv_workspace *workSpace;
	workSpace = gsl_eigen_symmv_alloc(dimension);

	//filter non-positive eigenvalues
	gsl_eigen_symmv(originalMatrix, eigenValues, eigenVectors, workSpace);
	gsl_eigen_symmv_sort(eigenValues, eigenVectors, GSL_EIGEN_SORT_VAL_DESC);

	std::cout << "eigen values: ";
	for (int i = 0; i < dimension; i++){
		std::cout << gsl_vector_get(eigenValues, i) << " ";
	}
	std::cout << std::endl;


	for (int i = 0; i < dimension; i++){
		if (i >= 10){
			gsl_vector_set(eigenValues, i, 0);
		}else if (gsl_vector_get(eigenValues, i) <= criteria){
			gsl_vector_set(eigenValues, i, criteria);
		}
	}

	std::cout << "eigen values: ";
	for (int i = 0; i < dimension; i++){
		std::cout << gsl_vector_get(eigenValues, i) << " ";
	}
	std::cout << std::endl;

	//recompute regularized matrix
	gsl_matrix_set_zero(diagonalizedEigenValues);
	for (int i = 0; i < dimension; i++){
		gsl_matrix_set(diagonalizedEigenValues, i, i, gsl_vector_get(eigenValues, i));
	}

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, eigenVectors, diagonalizedEigenValues, 0, workSpaceMatrix);
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, workSpaceMatrix, eigenVectors, 0, originalMatrix);

	for (int i = 0; i < dimension; i++){
		for (int j = 0; j< i+1; j++){
			SPDMatrix[dimension*i+j] = gsl_matrix_get(originalMatrix, i, j);
			if (i != j){
				SPDMatrix[dimension*j+i] = gsl_matrix_get(originalMatrix, i, j);
			}
		}
	}

	// std::cout << "SPD hessianMatrix: " << std::endl;
	// for (int i = 0; i < dimension; i++){
	// 	for (int j = 0; j< dimension; j++){
	// 		std::cout << SPDMatrix[dimension*i+j] << " ";			
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << std::endl;	

};

void newtonStep(double gradient[], double hessian[], int dimension, double newtonStepArray[]){
	gsl_vector* gradientVector = gsl_vector_alloc(dimension);
	gsl_matrix* hessianMatrix = gsl_matrix_alloc(dimension, dimension);
	gsl_vector* newtonStepVector = gsl_vector_alloc(dimension);

	std::cout << "gradient: ";
	for (int i = 0; i < dimension; i++){
		gsl_vector_set(gradientVector, i, gradient[i]);
		std::cout << gradient[i] << " ";
		for (int j = 0; j < dimension; j++){
			gsl_matrix_set(hessianMatrix, i, j, hessian[i*dimension+j]);
		}
	}
	std::cout << std::endl;

	gsl_linalg_cholesky_decomp1(hessianMatrix);
	gsl_linalg_cholesky_solve(hessianMatrix, gradientVector, newtonStepVector);

	for (int i = 0; i < dimension; i++){
		newtonStepArray[i] = gsl_vector_get(newtonStepVector, i);
	}
}
