#pragma once

#include <memory>
#include <random>

#include <mpi.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscds.h>
#include <iostream>

#include <ctime>
#include <chrono>

#include <numericalRecipes.h>

enum PRIOR_DISTRIBUTION{ UNIFORM, GAUSSIAN };

class NSE2dLagSolver{
private:
    DM          dm;
    TS          ts;
    Vec         u,r;
    PetscInt    timeSteps;
    PetscReal   time=0.0,tMax=1.0,deltaT,obs=0.0;
    // Vec         X,X_snap,intVecObs,intVecQoi;

    std::unique_ptr<double[]> samples;
    std::unique_ptr<double[]> ux;
    std::unique_ptr<double[]> vy;
    
    std::default_random_engine generator;
    std::normal_distribution<double> normalDistribution{0.0, 1.0};
    std::uniform_real_distribution<double> uniformDistribution{-1.0, 1.0};

public:
    MPI_Comm    comm;
    PetscInt    level;
    PetscInt    num_term;
    PetscReal   noiseVariance;

    NSE2dLagSolver(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_);
    ~NSE2dLagSolver(){
        VecDestroy(&u);
        VecDestroy(&r);
        // VecDestroy(&X);
        // VecDestroy(&X_snap);
        // VecDestroy(&intVecObs);
        // VecDestroy(&intVecQoi);
        TSDestroy(&ts);
        DMDestroy(&dm);
    };

    void updateGeneratorSeed(double seed_);
    void priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag=GAUSSIAN);

    void StabOptionOn();
    void SetupProblem();
    void LinearSystemSetup();
    void SolverSetup();
    void solve(bool flag=0);
//     double solve4Obs();
//     double solve4QoI();
//     double ObsOutput();
//     double QoiOutput();
//     double lnLikelihood();
};
