#pragma once

#include <memory>
#include <random>
#include <ctime>
#include <chrono>

#include <mpi.h>
#include <petsc.h>

#include <numericalRecipes.h>
#include <FEModule.h>

enum PRIOR_DISTRIBUTION{ UNIFORM, GAUSSIAN };

class NSE2dSolverCNDir{
private:
	Mat  LHS,LHSpA,J,Jt;
	Vec  Xn,Xn1,X_bar,X_snap,intVecObs,intVecQoi,RHS,r;
    SNES snes;
    KSP  ksp;
    PC   pc;

public:
    MPI_Comm           comm;
    structureMesh2D*   mesh;
    VortexDOF**        states;

	int stabOption=0,LHSIndicator=0;
	int level,timeSteps,num_term;
	double nu = 1;
	double time=0.0,tMax=1.0,deltaT;
	double obs,noiseVariance;
	double beta = 1.0;

    std::unique_ptr<double[]> samples;
    std::unique_ptr<double[]> ux;
    std::unique_ptr<double[]> vy;
    
	std::default_random_engine generator;
	std::normal_distribution<double> normalDistribution{0.0, 1.0};
    std::uniform_real_distribution<double> uniformDistribution{-1.0, 1.0};

    NSE2dSolverCNDir(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_, DMBoundaryType btype_=DM_BOUNDARY_NONE);
    ~NSE2dSolverCNDir(){
        delete mesh;
        MatDestroy(&LHS);
        MatDestroy(&LHSpA);
        MatDestroy(&J);
        MatDestroy(&Jt);
        VecDestroy(&Xn);
        VecDestroy(&Xn1);
        VecDestroy(&X_bar);
        VecDestroy(&X_snap);
        VecDestroy(&intVecObs);
        VecDestroy(&intVecQoi);
        VecDestroy(&RHS);
        VecDestroy(&r);

        SNESDestroy(&snes);
    };

    void updateGeneratorSeed(double seed_);
    void priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag=UNIFORM);

    void StabOptionOn();
    void LinearSystemSetup();
    void SolverSetup();
    void AttachNullspace(DM dmSol,Mat A);
    void ForwardStep();
    void solve(bool flag=0);
    double solve4Obs();
    double solve4QoI();
    double ObsOutput();
    double QoiOutput();
    double lnLikelihood();

	void FormFunction(SNES snes, Vec ss, Vec ff);
	void FormJacobian(SNES snes, Vec xx, Mat jac, Mat B);

	static PetscErrorCode FormFunctionStatic(SNES snes, Vec xx, Vec ff, void *ctx)
    {
		NSE2dSolverCNDir *ptr = static_cast<NSE2dSolverCNDir*>(ctx);
		ptr->FormFunction(snes, xx, ff);
		return 0;
	};

    static PetscErrorCode FormJacobianStatic(SNES snes, Vec xx, Mat jac, Mat B, void *ctx)
    {
        NSE2dSolverCNDir *ptr = static_cast<NSE2dSolverCNDir*>(ctx);
        ptr->FormJacobian(snes,xx,jac,B);
        return 0;
    };   

    void getValues(double x[], double y[], double *ux, double *uy, int size);
    void getGradients(double x[], double y[], int size);
    void getGradients(double x[], double y[], double uxx[], double uxy[], double uyx[], double uyy[], int size);
};
