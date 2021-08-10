#pragma once

#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>

#include <mpi.h>
#include <petsc.h>
#include <numericalRecipes.h>

enum ELEMENT_TYPE{ Q1, Q2, P1, P2 };

class Mixed2DMesh{
private:
public:
	MPI_Comm comm;

	int level;
	int vortex_num_per_row;
    int vortex_num_per_column;
    int total_element;
    int total_vortex;

	Mat M,G,D;
	Vec f;

	DM                     meshDM;
    ELEMENT_TYPE           etype;
    DMBoundaryType         btype;

	Mixed2DMesh(MPI_Comm comm_, int level_, ELEMENT_TYPE etype_=P1, DMBoundaryType btype_=DM_BOUNDARY_PERIODIC);
	~Mixed2DMesh(){
        MatDestroy(&M);
        MatDestroy(&G);
        MatDestroy(&D);
        VecDestroy(&f);
        DMDestroy(&meshDM);
    };
};

void GetElementCoordindates2D(DMDACoor2d **coords, int i, int j, double el_coords[]);

void ShapeFunctionP12D_Evaluate(double xi_p[], double Ni_p[]);

void ShapeFunctionP12D_Evaluate_dxi(double GNi[][4]);

void ShapeFunctionP12D_Evaluate_dx(double GNi[][4], double GNx[][4], double coords[], double *det_J);

void FormGradientOperatorP12D(double Ge1[], double Ge2[], double coords[]);

void FormScaledMassMatrixOperatorP12D(double Me1[],double Me2[],double coords[],double E,double nu[]);
void FormScaledMassMatrixOperatorP12D(double Me1[],double Me2[],double coords[],void(*Lame)(double,double,double[],double[],int), double samples[], int sampleSize);

void FormContinuityRhsP12D(double Fe1[],double Fe2[],double coords[],void(*Forcing)(double,double,double[],double[],int), double samples[], int sampleSize);

void FormIntegralOperatorP1(double Oe1[],double Oe2[],double coords[],double expCoef);

void AssembleM(Mat M,Mixed2DMesh* mesh,double nu,double Elasticity);
void AssembleM(Mat M,Mixed2DMesh* mesh,void(*Lame)(double,double,double[],double[],int),double samples[],int sampleSize);

void AssembleG(Mat G,Mixed2DMesh* mesh);

void AssembleF(Vec F,Mixed2DMesh* mesh,void(*Forcing)(double,double,double[],double[],int),double samples[],int sampleSize);

void AssembleIntegralOperator(Vec intVec,Mixed2DMesh* mesh,double expCoef);
void AssembleIntegralOperatorObs(Vec intVec,Mixed2DMesh* mesh,double expCoef);
void AssembleIntegralOperatorQoi(Vec intVec,Mixed2DMesh* mesh,double expCoef);
void AssembleIntegralOperatorQoi2(Vec intVec,Mixed2DMesh* mesh,double expCoef);