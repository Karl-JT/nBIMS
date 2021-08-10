#pragma once

#include <memory>
#include <random>

#include <mpi.h>
#include <petsc.h>
#include <ctime>
#include <chrono>

#include <numericalRecipes.h>
#include <FEModule.h>

enum PRIOR_DISTRIBUTION{ UNIFORM, GAUSSIAN };

class stokes2dSolver{
private:
    Mat LHS;
	Vec X,X_snap,intVecObs,intVecQoi,RHS;
    KSP ksp;
    PC  pc;

public:
    MPI_Comm           comm;
    structureMesh2D*   mesh;
    VortexDOF**        states;

	int stabOption=0;
	int level,timeSteps,num_term,rank;
	double nu = 0.1;
	double time=0.0,tMax=1.0,deltaT;
	double obs,noiseVariance;
	double beta = 1.0;

    std::unique_ptr<double[]> samples;
    std::unique_ptr<double[]> ux;
    std::unique_ptr<double[]> vy;
    
	std::default_random_engine generator;
	std::normal_distribution<double> normalDistribution{0.0, 1.0};
    std::uniform_real_distribution<double> uniformDistribution{-1.0, 1.0};

    stokes2dSolver(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_);
    ~stokes2dSolver(){
        delete mesh;
        MatDestroy(&LHS);
        VecDestroy(&X);
        VecDestroy(&intVecObs);
        VecDestroy(&intVecQoi);
        KSPDestroy(&ksp);
    };

    void updateGeneratorSeed(double seed_);
    void priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag=GAUSSIAN);

    void StabOptionOn();
    void LinearSystemSetup();
    void SolverSetup();
    void solve(bool flag=0);
    double solve4Obs();
    double solve4QoI();
    double ObsOutput();
    double QoiOutput();
    double lnLikelihood();
};
