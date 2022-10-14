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

class NSE2dSolverLag{
private:
    Mat LHS;
    Vec X,X_snap,intVecQoi,RHS;
    KSP ksp;
    PC  pc;
    IS  isrowcol;

public:
    MPI_Comm           comm;
    structureMesh2D*   mesh;
    VortexDOF**        states;

    int stabOption=0;
    int level,timeSteps,num_term,rank;
    int idx1, idx2;
    double nu=0.1;
    double time=0.0,tMax=1.0,deltaT;
    double noiseVariance;
    double beta = 1.0;
    double z0[10] = {0.49081423,0.90790136,0.01243684,0.31235,0.97549782,0.20306799,0.03477005,0.1107383,0.72502269,0.57489244};

    std::unique_ptr<double[]> obs_input;
    std::unique_ptr<double[]> z;
    std::unique_ptr<double[]> samples;
    std::unique_ptr<double[]> ux;
    std::unique_ptr<double[]> vy;
    
    std::default_random_engine generator;
    std::normal_distribution<double> normalDistribution{0.0, 1.0};
    std::uniform_real_distribution<double> uniformDistribution{-1.0, 1.0};

    NSE2dSolverLag(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_);
    ~NSE2dSolverLag(){
        delete mesh;
        MatDestroy(&LHS);
        VecDestroy(&X);
        VecDestroy(&X_snap);
        VecDestroy(&intVecQoi);
        VecDestroy(&RHS);
        KSPDestroy(&ksp);
    };

    void updateGeneratorSeed(double seed_);
    void priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag=UNIFORM);

    void StabOptionOn();
    void LinearSystemSetup();
    void SolverSetup();
    void ForwardStep();
    void UpdateZ(int time_idx);
    void solve(bool flag=0);
    double solve4QoI();
    double solve4Obs();
    void ObsOutput(double obs[], int size=1);
    double QoiOutput();
    double lnLikelihood();
};
