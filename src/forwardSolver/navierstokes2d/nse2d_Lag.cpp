#include "nse2d_Lag.h"
#include "nseforcing.h"

NSE2dSolverLag::NSE2dSolverLag(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_):comm(comm_), level(level_), num_term(num_term_), noiseVariance(noiseVariance_){
    mesh      = new structureMesh2D(comm_, level_, 3, Q1);
    timeSteps = std::pow(2, level_+1);
    deltaT    = tMax/timeSteps;

    double obs_read[20] = {0.63979175, 0.79484298, 0.17199418, 0.19541597, 1.13408598, 0.07899129, 0.17270835, -0.02744946, 0.84578063, 0.41139245, 1.11225036, 0.32293072, 0.65912191, -0.30282125, 1.62780146, -0.42925516, 0.65568777, -0.5237915, 1.35274024, -0.08863465};
    obs_input = std::make_unique<double[]>(20);
    samples   = std::make_unique<double[]>(num_term_);
    z         = std::make_unique<double[]>(10*(timeSteps+1));
    for (int i=0; i<10; i++){
        z[i] = z0[i];
        obs_input[2*i]   = obs_read[2*i];
        obs_input[2*i+1] = obs_read[2*i+1];
	//std::cout << z[i] << " ";
    };
    //std::cout << std::endl;

    DMCreateGlobalVector(mesh->meshDM,&X);
    DMCreateGlobalVector(mesh->meshDM,&intVecQoi);

    VecZeroEntries(X);
    VecDuplicate(X, &X_snap);
    VecDuplicate(X, &RHS);

    SolverSetup();
};

void NSE2dSolverLag::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};

void NSE2dSolverLag::LinearSystemSetup()
{
    AssembleM(mesh->M, mesh->meshDM);
    AssembleA(mesh->A, mesh->meshDM, nu);
    AssembleG(mesh->G, mesh->meshDM);
    AssembleQ(mesh->Q, mesh->meshDM);
    AssembleD(mesh->D, mesh->meshDM);

    AssembleIntegralOperator(intVecQoi,mesh->meshDM,0.5);

    Mat Workspace;
    MatScale(mesh->M, 1.0/deltaT);
    MatAXPY(mesh->G,1.0,mesh->Q, DIFFERENT_NONZERO_PATTERN);
    MatPtAP(mesh->G,mesh->D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Workspace);
    MatAXPY(mesh->A,1.0,mesh->M, SAME_NONZERO_PATTERN);
    MatAXPY(mesh->A,1.0,Workspace, DIFFERENT_NONZERO_PATTERN);

    MatDuplicate(mesh->A, MAT_DO_NOT_COPY_VALUES, &LHS);
    MatDestroy(&Workspace);
};

//Iterative solver
void NSE2dSolverLag::SolverSetup(){
    LinearSystemSetup();

    KSPCreate(comm, &ksp);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCFIELDSPLIT);
    PCFieldSplitSetDetectSaddlePoint(pc, PETSC_TRUE);
    //PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, NULL);
    //KSPMonitorSet(ksp, MyKSPMonitor, NULL, 0);
    PCSetFromOptions(pc);
    KSPSetTolerances(ksp, 1e-8, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetFromOptions(ksp);
    KSPSetPC(ksp, pc);
};

void NSE2dSolverLag::ForwardStep(){
    MatZeroEntries(mesh->C);
    VecZeroEntries(mesh->f);

    AssembleC(mesh->C,mesh->meshDM,X,mesh->l2gmapping);
	
    MatCopy(mesh->A,LHS,SAME_NONZERO_PATTERN);
    MatAXPY(LHS,1.0,mesh->C,SUBSET_NONZERO_PATTERN);

    AssembleF(mesh->f,mesh->meshDM,time,forcing,samples.get(),num_term);
    MatMultAdd(mesh->M, X, mesh->f, RHS);

    //MatCreateSubMatrix(LHS, isrowcol, isrowcol, MAT_REUSE_MATRIX, &LHS_sub);
    //MatView(LHS_sub, PETSC_VIEWER_STDOUT_WORLD);
    KSPSetOperators(ksp,LHS,LHS);
    //VecGetSubVector(RHS, isrowcol, &RHS_sub);
    KSPSolve(ksp, RHS, X);
    //VecRestoreSubVector(RHS, isrowcol, &RHS_sub);
};

void NSE2dSolverLag::UpdateZ(int time_idx){
    double pointwiseVel[2];
    double tracerLocation[2];
    for (int i=0; i<5; i++){
        tracerLocation[0] = z[10*time_idx+2*i];
        tracerLocation[1] = z[10*time_idx+2*i+1];
	//std::cout << "Last Location: " << tracerLocation[0] << " " << tracerLocation[1] <<std::endl;
        SolutionPointWiseInterpolation(mesh->meshDM, mesh->vortex_num_per_row, X, tracerLocation, pointwiseVel);
        //std::cout << "current velocity: " << pointwiseVel[0] << " " << pointwiseVel[1] << " "; 
	z[10*(time_idx+1)+2*i] = z[10*time_idx+2*i] + pointwiseVel[0]*deltaT;
        z[10*(time_idx+1)+2*i+1] = z[10*time_idx+2*i+1] + pointwiseVel[1]*deltaT;
	while (z[10*(time_idx+1)+2*i] >= 1.0) {
		z[10*(time_idx+1)+2*i] = z[10*(time_idx+1)+2*i]-1.0;	
	}
        while (z[10*(time_idx+1)+2*i] < 0.0) {
                z[10*(time_idx+1)+2*i] = z[10*(time_idx+1)+2*i]+1.0;
        }	
	while (z[10*(time_idx+1)+2*i+1] >= 1.0) {
		z[10*(time_idx+1)+2*i+1] = z[10*(time_idx+1)+2*i+1]-1.0;	
	}
        while (z[10*(time_idx+1)+2*i+1] < 0.0) {
                z[10*(time_idx+1)+2*i+1] = z[10*(time_idx+1)+2*i+1]+1.0;
        }
        //std::cout << "updated location: " << z[10*(time_idx+1)+2*i] << " " << z[10*(time_idx+1)+2*i+1] << std::endl;   
    };
};

void NSE2dSolverLag::solve(bool flag)
{
	VecZeroEntries(X);
	time = 0.0;

	for (int i = 0; i < timeSteps; i++){
	    //std::cout << "#################" << " level " << level << ", step " << i+1 << " #################" << std::endl;
	    //std::clock_t c_start = std::clock();
            //auto wcts = std::chrono::system_clock::now();	
	    time = time+deltaT;
            //std::cout << "time " << time << std::endl;
	    ForwardStep();
            UpdateZ(i);

	    if (abs(time-0.5) <1e-6){
                idx1=i+1;	    
            }	
            //std::clock_t c_end = std::clock();
            //double time_elapsed_ms = (c_end-c_start)/ (double)CLOCKS_PER_SEC;
            //std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
            //std::cout << "wall time " << wctduration.count() << " cpu  time: " << time_elapsed_ms << std::endl;	
	}
        VecCopy(X, X_snap);
        idx2 = timeSteps;
};

void NSE2dSolverLag::priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag)
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
};

double NSE2dSolverLag::solve4QoI()
{
    double qoi=QoiOutput();
    return qoi;
};

double NSE2dSolverLag::solve4Obs(){
    double obs[20];
    ObsOutput(obs, 10);
    return 0;
};

void NSE2dSolverLag::ObsOutput(double obs[], int size){
    for (int i=0; i<size; i++){
        obs[i] = z[10*idx1+i];
        obs[10+i] = z[10*idx2+i];
    }
};

double NSE2dSolverLag::QoiOutput(){
    double qoi;
    VecDot(intVecQoi, X, &qoi);
    qoi = 100.*qoi;
 //   std::cout << "qoi: " << qoi << std::endl << std::flush;
    return qoi;
};

double NSE2dSolverLag::lnLikelihood(){
	double obsResult[20];
        ObsOutput(obsResult, 10);
	double lnLikelihood = 0;
        for (int i=0; i<20; i++){
            lnLikelihood+=-0.5/noiseVariance*pow(obsResult[i]-obs_input[i],2);
        }
//        std::cout << "lnlikelihood: " << lnLikelihood << std::endl << std::flush;
	return lnLikelihood;
};
