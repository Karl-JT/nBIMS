#include "stokes2dpenalty.h"

PetscErrorCode MyKSPMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *ctx)
{
	PetscPrintf(PETSC_COMM_SELF, "iteration %D KSP Residual norm %14.12e \n", n, rnorm);
	return 0;
}

static double viscNu(double x, double y, double samples[], int sampleSize){
    double output = 1.5+samples[0]*(std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y));
    //double output = exp(samples[0]*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y));
    return output;
};

static double penalty(double x, double y, double samples[], int sampleSize){
    double output = 1.0/100.0;//(20.0+samples[0]*(std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y)));
    //double output = 1.0/(10.0*viscNu(x,y,samples,sampleSize)+exp(samples[0]*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y)));
    return output;
};

static void forcing(double x, double y, double time, double output[], double samples[], int sampleSize){
    output[0] = 2000*std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y);
    output[1] = -2000*std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y);
};

stokes2dpenaltySolver::stokes2dpenaltySolver(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_):comm(comm_), level(level_), num_term(num_term_), noiseVariance(noiseVariance_){
    mesh      = new structureMesh2D(comm_, level_, 3, Q1);
    // obs       = 1.774282582683544;
    obs       = 0.62859485698;//3.16811610103525;//0.93338859565;

	samples   = std::make_unique<double[]>(num_term_);

	DMCreateGlobalVector(mesh->meshDM,&X);
	DMCreateGlobalVector(mesh->meshDM,&intVecObs);
	DMCreateGlobalVector(mesh->meshDM,&intVecQoi);

    VecZeroEntries(X);
    SolverSetup();
};

void stokes2dpenaltySolver::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};


void stokes2dpenaltySolver::LinearSystemSetup()
{
    AssembleP(mesh->P, mesh->meshDM, penalty, samples.get(), num_term);
    AssemblenuA(mesh->A, mesh->meshDM, viscNu, samples.get(), num_term);
    AssembleG(mesh->G, mesh->meshDM);
    AssembleQ(mesh->Q, mesh->meshDM);
    AssembleD(mesh->D, mesh->meshDM);
    AssembleF(mesh->f,mesh->meshDM,1,forcing,samples.get(),num_term);

    AssembleIntegralOperator(intVecQoi,mesh->meshDM,1.5);
    AssembleIntegralOperator(intVecObs,mesh->meshDM,0.5);

    MatAXPY(mesh->G,1.0,mesh->Q, DIFFERENT_NONZERO_PATTERN);
    MatPtAP(mesh->G,mesh->D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&mesh->M);

    MatPtAP(mesh->P,mesh->D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&PQ);
    MatDuplicate(mesh->A, MAT_DO_NOT_COPY_VALUES, &LHS);
    MatAXPY(LHS, 1.0, mesh->M, DIFFERENT_NONZERO_PATTERN);
    MatAXPY(LHS, -1.0, PQ, DIFFERENT_NONZERO_PATTERN);
}

//Iterative solver
void stokes2dpenaltySolver::SolverSetup(){
    LinearSystemSetup();

	KSPCreate(comm, &ksp);
	KSPGetPC(ksp, &pc);

	PCSetType(pc, PCFIELDSPLIT);
    const PetscInt ufields[] = {0,1},pfields[1] = {2};
    KSPGetPC(ksp,&pc);
    PCFieldSplitSetBlockSize(pc,3);
    PCFieldSplitSetFields(pc,"u",2,ufields,ufields);
    PCFieldSplitSetFields(pc,"p",1,pfields,pfields);    

	// KSPMonitorSet(ksp, MyKSPMonitor, NULL, 0);
	KSPSetTolerances(ksp, 1e-8, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetFromOptions(ksp);
	KSPSetPC(ksp, pc);
}

void stokes2dpenaltySolver::solve(bool flag)
{
    MatZeroEntries(mesh->A);
    MatZeroEntries(mesh->P);
    AssemblenuA(mesh->A, mesh->meshDM, viscNu, samples.get(), num_term);
    AssembleP(mesh->P, mesh->meshDM, penalty, samples.get(), num_term);
    MatPtAP(mesh->P,mesh->D,MAT_REUSE_MATRIX,PETSC_DEFAULT,&PQ);
    // MatView(mesh->A, PETSC_VIEWER_STDOUT_WORLD);

    MatCopy(mesh->M,LHS,SUBSET_NONZERO_PATTERN);
    MatAXPY(LHS,1.0,mesh->A,SUBSET_NONZERO_PATTERN);
    MatAXPY(LHS,1.0,PQ,SUBSET_NONZERO_PATTERN);
    // MatView(LHS, PETSC_VIEWER_STDOUT_WORLD);

	KSPSetOperators(ksp,LHS,LHS);
	KSPSolve(ksp, mesh->f, X);
    // VecView(X, PETSC_VIEWER_STDOUT_WORLD);

    solve4QoI();
	solve4Obs();
};

void stokes2dpenaltySolver::priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag)
{
	switch(flag){
        case UNIFORM:
            for (int i = 0; i<num_term; ++i){
                initialSamples[i] = uniformDistribution(generator);
            }
            break;
        case GAUSSIAN:
            for (int i = 0; i<num_term; ++i){
                initialSamples[0] = normalDistribution(generator);
            }
            break;
        default:
            for (int i = 0; i<num_term; ++i){
                initialSamples[i] = uniformDistribution(generator);
            }
    }
}

double stokes2dpenaltySolver::solve4QoI(){
    return QoiOutput();
};

double stokes2dpenaltySolver::solve4Obs(){
    return ObsOutput();
};

double stokes2dpenaltySolver::ObsOutput(){
    double obs;
    VecDot(intVecObs, X, &obs);
    obs = 1.*obs;
    return obs;	
}

double stokes2dpenaltySolver::QoiOutput(){
    double qoi;
    VecDot(intVecQoi, X, &qoi);
    qoi = 1.*qoi;
    return qoi;
}

double stokes2dpenaltySolver::lnLikelihood(){
	double obsResult = ObsOutput();
	double lnLikelihood = -0.5/noiseVariance*pow(obsResult-obs,2);
	return lnLikelihood;
}
