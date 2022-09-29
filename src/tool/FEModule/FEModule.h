#pragma once

#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>

#include <mpi.h>
#include <petsc.h>
#include <numericalRecipes.h>

enum ELEMENT_TYPE{ Q1, Q2, P1, P2 };

typedef struct
{
    double u;
    double v;
    double p;
} VortexDOF;

class structureMesh2D{
private:
public:
	MPI_Comm comm;

	int level;
	int dof;
	int vortex_num_per_row;
    int vortex_num_per_column;
	int ngp; 

	Mat M,A,G,Q,C,J,D,P;
	Vec f;

	DM                     meshDM;
    ELEMENT_TYPE           etype;
    DMBoundaryType         btype;
    ISLocalToGlobalMapping l2gmapping;
    IS                     bottomUx,bottomUy,leftUx,leftUy,rightUx,rightUy,topUx,topUy;

	structureMesh2D(MPI_Comm comm_, int level_, int dof_, ELEMENT_TYPE etype_, DMBoundaryType btype_=DM_BOUNDARY_PERIODIC);
	~structureMesh2D(){
        MatDestroy(&M);
        MatDestroy(&A);
        MatDestroy(&G);
        MatDestroy(&Q);
        MatDestroy(&C);
        MatDestroy(&J);
        MatDestroy(&D);
        MatDestroy(&P);
        VecDestroy(&f);
        DMDestroy(&meshDM);

        switch (btype)
        {
            case DM_BOUNDARY_PERIODIC:
            {
                break;
            }

            case DM_BOUNDARY_NONE:
            {
                ISDestroy(&bottomUx);
                ISDestroy(&bottomUy);
                ISDestroy(&leftUx);
                ISDestroy(&leftUy);
                ISDestroy(&rightUx);
                ISDestroy(&rightUy);
                ISDestroy(&topUx);
                ISDestroy(&topUy);
                break;
            }

            default:
            {
                break;
            }
        }
    };
};

void GetElementCoordindates2D(DMDACoor2d **coords, int i, int j, double el_coords[]);

void ConstructGaussQuadrature2D(int ngp, double gp_xi[][2], double gp_weight[]);

void ShapeFunctionQ12D_Evaluate(double xi_p[], double Ni_p[]);

void ShapeFunctionQ12D_Evaluate_dxi(double xi_p[], double GNi[][4]);

void ShapeFunctionQ12D_Evaluate_dx(double GNi[][4], double GNx[][4], double coords[], double *det_J);

void FormStressOperatorQ1(double Kex[],double Key[], double coords[], double eta[]);
void FormStressOperatorQ1nu(double Kex[],double Key[], double coords[], double(*Visc)(double, double, double[], int), double samples[], int sampleSize);

int ASS_MAP_wIwDI_uJuDJ(int wi,int wd,int w_NPE,int w_dof,int ui,int ud,int u_NPE,int u_dof);

void FormGradientOperatorQ12D(double Ge[], double coords[]);

void FormAdvectOperatorQ12D(double Ce[], double coords[], double vel[2][4], int m, int n);

void FormAdvectJacobian(double CJ[], double coords[], double vel[2][4], int m, int n);

void GetExplicitVel(VortexDOF **states, int i, int j, double vel[2][4]);

void FormScaledMassMatrixOperatorQ12D(double Me[],double coords[],double eta[]);

void FormMomentumRhsQ12D(double Fe[],double coords[],double t,void(*Forcing)(double, double, double, double[], double[], int), double samples[], int sampleSize);

void FormContinuityRhsQ12D(double Fe[],double coords[],double hc[]);

void DMDAGetElementEqnums2D_up(MatStencil s_u[],MatStencil s_p[],int i,int j);

void FormPreconditionerQ1isoQ2(double Qe[],int i, int j);

void FormIntegralOperator(double Oe[], double coords[], double expCoef1);
void FormIntegralOperator2(double Oe[], double coords[], double expCoef1);

void AssembleM(Mat M, DM meshDM);

void AssembleP(Mat P, DM meshDM, double(*penalty)(double,double,double[],int), double samples[], int sampleSize);

void AssembleA(Mat A, DM meshDM, double nu);
void AssemblenuA(Mat A, DM meshDM, double(*Visc)(double, double, double[], int), double samples[], int sampleSize);

void AssembleG(Mat G, DM meshDM);

void AssembleQ(Mat Q, DM meshDM);

void AssembleC(Mat C, DM meshDM, Vec X, ISLocalToGlobalMapping l2gmapping);

void AssembleJ(Mat J, DM meshDM, Vec X, ISLocalToGlobalMapping l2gmapping);

void AssembleD(Mat D, DM meshDM);

void DMDASetValuesLocalStencil2D_ADD_VALUES(VortexDOF **fields_F,MatStencil u_eqn[],MatStencil p_eqn[],double Fe_u[],double Fe_p[]);

void AssembleF(Vec F,DM meshDM,double t,void(*Forcing)(double, double, double, double[], double[], int),double samples[], int sampleSize);

void AssembleIntegralOperator(Vec intVec,DM meshDM, double expCoef1);
void AssembleIntegralOperator2(Vec intVec,DM meshDM, double expCoef1);

void ApplyBoundaryCondition(Mat Sys, Vec Sol, Vec Rhs, IS boundaryIS);

void Interpolate(Mat M, Vec load, Vec interpolation);

void SolutionPointWiseInterpolation(DM meshDM, int votex_num_per_row, Vec X, double z[], double pointwiseVel[]);
