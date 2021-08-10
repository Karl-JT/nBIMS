#pragma once

#include <iostream>
#include <fstream>
#include <petsc.h>
#include <string>
#include <sstream>
#include <chrono>
#include <memory>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <mpi.h>

typedef struct{
	double gp_coords[50];
	double f[25];
} GaussPointCoefficients;

typedef struct
{
	double u_dof;
} StokesDOF;

typedef struct _p_CellProperties *CellProperties;
struct _p_CellProperties{
	int ncells;
	int mx, my;
	int sex, sey;
	GaussPointCoefficients *gpc;
};


typedef struct _p_Mesh* Mesh;
struct _p_Mesh{
	int level;
	int vortex_num_per_row;
	int sex, sey, Imx, Jmy;
	int gauss_points = 25; 

	MPI_Comm mpicomm;
	DM   meshDM;
	DM   cda;
	Vec  coords;
	DMDACoor2d **_coords;
	CellProperties cell_properties;
}

void gauleg(int n, double x[], double w[]);
PetscErrorCode init();
int CellPropertiesCreate(DM meshDM, CellProperties *cell);
int CellPropertiesDestroy(CellProperties *cell);
void CellPropertiesGetCell(CellProperties C, int II, int J, GaussPointCoefficients **G);
void GetElementCoords(DMDACoor2d **coords, int i, int j, double el_coords[]);
void ConstructGaussQuadrature2D(double gp_xi[][2], double gp_weight[]);
void ShapeFunctionQ1Evaluate(double _xi[], double Ni[]);
void ShapeFunctionQ1Evaluate_dxi(double _xi[], double GNi[][4]);
void ShapeFunctionQ1Evaluate_dx(double GNi[][4], double GNx[][4], double coords[], double *det_J);
void FormStressOperatorQ1(double Ke[], double coords[]);
void FormGradientOperatorQ1(double Ge[], double coords[], int dir);
void FormQ1isoQ2Precond(double De[], int location);
void AssembleA(Mat A, DM meshDM, CellProperties cell_properties);
void AssembleB(Mat B, DM meshDM, CellProperties cell_properties, int dir);
void AssembleD(Mat D, DM meshDM, CellProperties cell_properties);
void AssembleF(Vec F, DM meshDM, CellProperties cell_properties);
void DMDAGetElementEqnums(MatStencil s[], int i, int j);
void FormMomentumRhsQ1(double Fe[],double coords[],double f[]);
void VecSetValuesStencil(StokesDOF **fields_F,MatStencil sqrStencil[], double Fe_u[]);


// class structMesh{
// private:
// public:
// 	int level;
// 	int vortex_num_per_row;
// 	int sex, sey, Imx, Jmy;
// 	int gauss_points = 25; 

// 	MPI_Comm mpicomm;
// 	DM   meshDM;
// 	DM   cda;
// 	Vec  coords;
// 	DMDACoor2d **_coords;
// 	CellProperties cell_properties;

// 	structMesh(int level_, MPI_Comm mpicomm_):level(level_), mpicomm(mpicomm_){};
// 	~structMesh(){};

// 	PetscErrorCode init();
// 	int CellPropertiesCreate(DM meshDM, CellProperties *cell);
// 	int CellPropertiesDestroy(CellProperties *cell);
// 	void CellPropertiesGetCell(CellProperties C, int II, int J, GaussPointCoefficients **G);
// 	void GetElementCoords(DMDACoor2d **coords, int i, int j, double el_coords[]);
// 	void ConstructGaussQuadrature2D(double gp_xi[][2], double gp_weight[]);
// 	void ShapeFunctionQ1Evaluate(double _xi[], double Ni[]);
// 	void ShapeFunctionQ1Evaluate_dxi(double _xi[], double GNi[][4]);
// 	void ShapeFunctionQ1Evaluate_dx(double GNi[][4], double GNx[][4], double coords[], double *det_J);
// 	void FormStressOperatorQ1(double Ke[], double coords[]);
// 	void FormGradientOperatorQ1(double Ge[], double coords[], int dir);
// 	void FormQ1isoQ2Precond(double De[], int location);
// 	void AssembleA(Mat A, DM meshDM, CellProperties cell_properties);
// 	void AssembleB(Mat B, DM meshDM, CellProperties cell_properties, int dir);
// 	void AssembleD(Mat D, DM meshDM, CellProperties cell_properties);
// 	void AssembleF(Vec F, DM meshDM, CellProperties cell_properties);
// 	void DMDAGetElementEqnums(MatStencil s[], int i, int j);
// 	void FormMomentumRhsQ1(double Fe[],double coords[],double f[]);
// 	void VecSetValuesStencil(StokesDOF **fields_F,MatStencil sqrStencil[], double Fe_u[]);
// };




// structMesh(int level_, MPI_Comm mpicomm_):level(level_), mpicomm(mpicomm_){};
// ~structMesh(){};