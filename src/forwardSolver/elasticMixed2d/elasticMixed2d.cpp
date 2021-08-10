#include "elasticMixed2d.h"

PetscErrorCode MyKSPMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *ctx)
{
	PetscPrintf(PETSC_COMM_SELF, "iteration %D KSP Residual norm %14.12e \n", n, rnorm);
	return 0;
}

static void Lame(double x,double y,double lameC[],double samples[],int sampleSize){
    lameC[0] = exp(samples[0]*(std::cos(2.*M_PI*x)*std::sin(2.*M_PI*y))); //mu
    lameC[1] = exp(samples[0]*(std::sin(2.*M_PI*x)*std::cos(2.*M_PI*y))); //lambda
};

static void forcing(double x, double y, double output[], double samples[], int sampleSize){
    output[0] = 1000*std::sin(2.*M_PI*x);
    output[1] = 1000*std::sin(2.*M_PI*y);
};

elasticMixed2dSolver::elasticMixed2dSolver(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_):comm(comm_), level(level_), num_term(num_term_), noiseVariance(noiseVariance_){
    mesh      = new Mixed2DMesh(comm_, level_);
    obs       = -9.024514268027048; //-0.506750014397834; //15.562160632; //3.507886091459774;

	samples   = std::make_unique<double[]>(num_term_);

    VecDuplicate(mesh->f,&X);
    VecDuplicate(mesh->f,&intVecObs);
    VecDuplicate(mesh->f,&intVecQoi);

    VecZeroEntries(X);
    VecZeroEntries(intVecObs);
    VecZeroEntries(intVecQoi);
    // VecZeroEntries(intVecQoi2);

    SolverSetup();
};

void elasticMixed2dSolver::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};

void elasticMixed2dSolver::LinearSystemSetup()
{
    // for (int i = 0; i < 3*mesh->total_element+2*mesh->total_vortex; i++){
    //     MatSetValue(mesh->M, i, i, 0, INSERT_VALUES);
    // }
    // MatAssemblyBegin(mesh->M,  MAT_FLUSH_ASSEMBLY);
    // MatAssemblyEnd(mesh->M,  MAT_FLUSH_ASSEMBLY);

    AssembleM(mesh->M,mesh,Lame,samples.get(),num_term);
    AssembleG(mesh->G,mesh);
    MatTranspose(mesh->G,MAT_INITIAL_MATRIX,&mesh->D);

    AssembleF(mesh->f,mesh,forcing,samples.get(),num_term);
    VecScale(mesh->f, -1.0);

    VecZeroEntries(intVecQoi);
    VecZeroEntries(intVecObs);

    AssembleIntegralOperator(intVecObs,mesh,0.5);
    // AssembleIntegralOperatorObs(intVecObs,mesh,0.5);
    AssembleIntegralOperatorQoi2(intVecQoi,mesh,0.5);

    MatDuplicate(mesh->M,MAT_COPY_VALUES,&LHS);
    MatAXPY(LHS,-1.0,mesh->D,DIFFERENT_NONZERO_PATTERN);
    MatAXPY(LHS,-1.0,mesh->G,DIFFERENT_NONZERO_PATTERN);
}

//Iterative solver
void elasticMixed2dSolver::SolverSetup(){
    LinearSystemSetup();

	KSPCreate(comm, &ksp);
	KSPGetPC(ksp, &pc);
    PCSetFromOptions(pc);
	// KSPSetTolerances(ksp, 1e-8, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT);
	// KSPMonitorSet(ksp, MyKSPMonitor, NULL, 0);
	KSPSetFromOptions(ksp);
	KSPSetPC(ksp, pc);
}

void elasticMixed2dSolver::solve(bool flag)
{
    MatZeroEntries(mesh->M);
    AssembleM(mesh->M,mesh,Lame,samples.get(),num_term);

    MatCopy(mesh->M,LHS,SUBSET_NONZERO_PATTERN);
    MatAXPY(LHS,-1.0,mesh->G,SUBSET_NONZERO_PATTERN);
    MatAXPY(LHS,-1.0,mesh->D,SUBSET_NONZERO_PATTERN);

    // MatView(LHS, PETSC_VIEWER_STDOUT_WORLD);
	KSPSetOperators(ksp,LHS,LHS);

    // std::clock_t c_start = std::clock();
    // auto wcts = std::chrono::system_clock::now();

	KSPSolve(ksp, mesh->f, X);

    // std::clock_t c_end = std::clock();
    // double time_elapsed_ms = (c_end-c_start)/ (double)CLOCKS_PER_SEC;
    // std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
    // std::cout << "wall time " << wctduration.count() << " cpu  time: " << time_elapsed_ms << std::endl;	

    solve4QoI();
	solve4Obs();
};

void elasticMixed2dSolver::priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag)
{
	switch(flag){
        case UNIFORM:
            initialSamples[0] = uniformDistribution(generator);
            break;
        case GAUSSIAN:
            initialSamples[0] = normalDistribution(generator);
            break;
        default:
            initialSamples[0] = uniformDistribution(generator);
    }
}

double elasticMixed2dSolver::solve4QoI(){
    return QoiOutput();
};

double elasticMixed2dSolver::solve4Obs(){
    return ObsOutput();
};

double elasticMixed2dSolver::ObsOutput(){
    double obs;
    VecDot(intVecObs, X, &obs);
    obs = 1.*obs;
    return obs;	
}

double elasticMixed2dSolver::QoiOutput(){
    double qoi;
    VecDot(intVecQoi, X, &qoi);
    qoi = 1.*qoi;
    return qoi;
}

double elasticMixed2dSolver::lnLikelihood(){
	double obsResult = ObsOutput();
	double lnLikelihood = -0.5/noiseVariance*pow(obsResult-obs,2);
    // std::cout << samples[0] << " " << obsResult << " " << obs << " " << lnLikelihood << std::endl;
	return lnLikelihood;
}
