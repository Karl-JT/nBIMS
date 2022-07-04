#pragma once

#include <memory>
#include <random>

#include <ctime>
#include <chrono>

#include <numericalRecipes.h>
#include <FEModule2dMixed.h>

enum PRIOR_DISTRIBUTION{ UNIFORM, GAUSSIAN };

class elasticMixed2dSolver{
private:
    Mat LHS;
    Vec X,intVecObs,intVecQoi,RHS;
    KSP ksp;
    PC  pc;

public:
    MPI_Comm           comm;
    Mixed2DMesh*       mesh;

    int stabOption=0;
    int level,num_term,rank;
    //double nu = 1.0;
    //double Elasticity=1e11;
    double obs,noiseVariance;
    double beta = 1.0;

    std::unique_ptr<double[]> samples;
    
    std::default_random_engine generator;
    std::normal_distribution<double> normalDistribution{0.0, 1.0};
    std::uniform_real_distribution<double> uniformDistribution{-1.0, 1.0};

    elasticMixed2dSolver(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_);
    ~elasticMixed2dSolver(){
        delete mesh;
        MatDestroy(&LHS);
        VecDestroy(&X);
        VecDestroy(&intVecObs);
        VecDestroy(&intVecQoi);
        KSPDestroy(&ksp);
    };

    void updateGeneratorSeed(double seed_);
    void priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag=GAUSSIAN);

    void LinearSystemSetup();
    void SolverSetup();
    void solve(bool flag=0);
    double solve4Obs();
    double solve4QoI();
    double ObsOutput();
    double QoiOutput();
    double lnLikelihood();
};
