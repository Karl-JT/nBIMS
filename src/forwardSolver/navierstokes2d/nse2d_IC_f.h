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

class NSE2dSolverIC{
private:
    Mat LHS,M;
	Vec X,X_snap,intVecObs,intVecQoi,RHS;
    KSP ksp,ksp_int;
    PC  pc,pc_int;

public:
    MPI_Comm           comm;
    structureMesh2D*   mesh;
    VortexDOF**        states;

	int stabOption=0;
	int level,timeSteps,num_term;
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

    NSE2dSolverIC(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_);
    ~NSE2dSolverIC(){
        delete mesh;
        MatDestroy(&LHS);
        MatDestroy(&M);
        VecDestroy(&X);
        VecDestroy(&X_snap);
        VecDestroy(&intVecObs);
        VecDestroy(&intVecQoi);
        VecDestroy(&RHS);
        KSPDestroy(&ksp);
        KSPDestroy(&ksp_int);
    };

    void updateGeneratorSeed(double seed_);
    void priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag=UNIFORM);

    void StabOptionOn();
    void LinearSystemSetup();
    void InterpolateIC();
    void SolverSetup();
    void ForwardStep();
    void solve(bool flag=0);
    double solve4Obs();
    double solve4QoI();
    double ObsOutput();
    double QoiOutput();
    double lnLikelihood();
};
