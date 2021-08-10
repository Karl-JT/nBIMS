#include "nse2d_IC_f.h"
#include "nseIC.h"
#include "nseforcing.h"

NSE2dSolverIC::NSE2dSolverIC(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_):comm(comm_), level(level_), num_term(num_term_), noiseVariance(noiseVariance_){
    mesh      = new structureMesh2D(comm_, level_, 3, Q1);
    obs       = -0.965649652758160;
    timeSteps = std::pow(2, level_+1);
	deltaT    = tMax/timeSteps;

	samples   = std::make_unique<double[]>(num_term_);

	DMCreateGlobalVector(mesh->meshDM,&X);
	DMCreateGlobalVector(mesh->meshDM,&intVecObs);
	DMCreateGlobalVector(mesh->meshDM,&intVecQoi);

    VecZeroEntries(X);
    VecDuplicate(X, &X_snap);
    VecDuplicate(X, &RHS);

    SolverSetup();
};

void NSE2dSolverIC::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};

void NSE2dSolverIC::LinearSystemSetup()
{
    AssembleM(mesh->M, mesh->meshDM);
    AssembleA(mesh->A, mesh->meshDM, nu);
    AssembleG(mesh->G, mesh->meshDM);
    AssembleQ(mesh->Q, mesh->meshDM);
    AssembleD(mesh->D, mesh->meshDM);

    Mat Workspace;
    MatDuplicate(mesh->M, MAT_COPY_VALUES, &M);
    MatScale(mesh->M, 1.0/deltaT);
    MatAXPY(mesh->G,1.0,mesh->Q, DIFFERENT_NONZERO_PATTERN);
    MatPtAP(mesh->G,mesh->D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Workspace);
    MatAXPY(mesh->A,1.0,mesh->M, SAME_NONZERO_PATTERN);
    MatAXPY(mesh->A,1.0,Workspace, DIFFERENT_NONZERO_PATTERN);

    MatDuplicate(mesh->A, MAT_DO_NOT_COPY_VALUES, &LHS);
    MatDestroy(&Workspace);
}

void NSE2dSolverIC::InterpolateIC(){
    AssembleF(RHS,mesh->meshDM,0.0,IC,samples.get(),num_term);
	KSPSolve(ksp_int, RHS, X);    
    VecView(X, PETSC_VIEWER_STDOUT_WORLD);
}

//Iterative solver
void NSE2dSolverIC::SolverSetup(){
    LinearSystemSetup();

	KSPCreate(PETSC_COMM_SELF, &ksp_int);
	KSPSetType(ksp_int, KSPGMRES);
	KSPSetOperators(ksp_int, M, M);
	KSPGetPC(ksp_int, &pc_int);
	PCSetType(pc_int, PCJACOBI);
	PCSetUp(pc_int);
    KSPSetUp(ksp_int);

	KSPCreate(comm, &ksp);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCFIELDSPLIT);
	PCFieldSplitSetDetectSaddlePoint(pc, PETSC_TRUE);
	PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, NULL);
	// KSPMonitorSet(ksp, MyKSPMonitor, NULL, 0);
	KSPSetTolerances(ksp, 1e-8, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetFromOptions(ksp);
	KSPSetPC(ksp, pc);
}

void NSE2dSolverIC::ForwardStep(){
    MatZeroEntries(mesh->C);
    VecZeroEntries(mesh->f);

    AssembleC(mesh->C,mesh->meshDM,X,mesh->l2gmapping);
	
    MatCopy(mesh->A,LHS,SAME_NONZERO_PATTERN);
    MatAXPY(LHS,1.0,mesh->C,SUBSET_NONZERO_PATTERN);

    AssembleF(mesh->f,mesh->meshDM,time,forcing,samples.get(),num_term);
    MatMultAdd(mesh->M, X, mesh->f, RHS);

	KSPSetOperators(ksp,LHS,LHS);
	KSPSolve(ksp, RHS, X);
};

void NSE2dSolverIC::solve(bool flag)
{
	VecZeroEntries(X);
    InterpolateIC();
	time = 0.0;

	for (int i = 0; i < timeSteps; ++i){
		// std::cout << "#################" << " level " << level << ", step " << i+1 << " #################" << std::endl;
		// std::clock_t c_start = std::clock();
		// auto wcts = std::chrono::system_clock::now();	

		time = time+deltaT;
		ForwardStep();

		if (abs(time-0.5) <1e-6){
            VecCopy(X, X_snap);
            solve4QoI();
			if (flag == 1){
				return;
			}
		}	
    // std::clock_t c_end = std::clock();
    // double time_elapsed_ms = (c_end-c_start)/ (double)CLOCKS_PER_SEC;
    // std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
    // std::cout << "wall time " << wctduration.count() << " cpu  time: " << time_elapsed_ms << std::endl;	
	}
	solve4Obs();
};

void NSE2dSolverIC::priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag)
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

double NSE2dSolverIC::solve4QoI(){
    AssembleIntegralOperator(intVecQoi,mesh->meshDM,1.5);
    return QoiOutput();
};

double NSE2dSolverIC::solve4Obs(){
    AssembleIntegralOperator(intVecObs,mesh->meshDM,0.5);
    return ObsOutput();
};

double NSE2dSolverIC::ObsOutput(){
    double obs;
    VecDot(intVecObs, X, &obs);
    obs = 100.*obs;
    return obs;	
}

double NSE2dSolverIC::QoiOutput(){
    double qoi;
    VecDot(intVecQoi, X_snap, &qoi);
    qoi = 100.*qoi;
    return qoi;
}

double NSE2dSolverIC::lnLikelihood(){
	double obsResult = ObsOutput();
	double lnLikelihood = -0.5/noiseVariance*pow(obsResult-obs,2);
	return lnLikelihood;
}
