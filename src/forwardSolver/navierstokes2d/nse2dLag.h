#pragma once

#include <memory>
#include <random>

#include <mpi.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscds.h>

#include <ctime>
#include <chrono>

#include <numericalRecipes.h>

enum PRIOR_DISTRIBUTION{ UNIFORM, GAUSSIAN };

class NSE2dSolver{
private:
    DM          dm;
    PetscFE	fe[2];
    TS          ts;
    Vec         u,r;

public:
    MPI_Comm    comm;
    PetscInt    level;
    PetscInt    num_term;
    PetscReal   noiseVariance;

    NSE2dSolver(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_);
    ~NSE2dSolver(){
	PetscFEDestroy(&fe[0]);
	PetscFEDestroy(&fe[1]);

	VecDestroy(&u);
	VecDestroy(&r);
	TSDestroy(&ts);
	DMDestroy(&dm);
    };

    void updateGeneratorSeed(double seed_);
    void priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag=GAUSSIAN);

    void StabOptionOn();
    void LinearSystemSetup();
    void SolverSetup();
    void ForwardStep();
    void solve(bool flag=0);
    double solve4Obs();
    double solve4QoI();
    double ObsOutput();
    double QoiOutput();
    double lnLikelihood();
};
