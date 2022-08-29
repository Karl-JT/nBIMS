#include "nse2d.h"
#include "nseforcing.h"

NSE2dSolver::NSE2dSolver(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_):comm(comm_), level(level_), num_term(num_term_), noiseVariance(noiseVariance_){
    mesh      = new structureMesh2D(comm_, level_, 3, Q1);
    obs       = 43.30310812; //-9.65649652758160; //-1.876536734487264; //
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

void NSE2dSolver::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};

void NSE2dSolver::LinearSystemSetup()
{
    AssembleM(mesh->M, mesh->meshDM);
    AssembleA(mesh->A, mesh->meshDM, nu);
    AssembleG(mesh->G, mesh->meshDM);
    AssembleQ(mesh->Q, mesh->meshDM);
    AssembleD(mesh->D, mesh->meshDM);

    AssembleIntegralOperator(intVecQoi,mesh->meshDM,1.5);
    AssembleIntegralOperator(intVecObs,mesh->meshDM,0.5);

    Mat Workspace;
    MatScale(mesh->M, 1.0/deltaT);
    MatAXPY(mesh->G,1.0,mesh->Q, DIFFERENT_NONZERO_PATTERN);
    MatPtAP(mesh->G,mesh->D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Workspace);
    MatAXPY(mesh->A,1.0,mesh->M, SAME_NONZERO_PATTERN);
    MatAXPY(mesh->A,1.0,Workspace, DIFFERENT_NONZERO_PATTERN);

    //int is_size = mesh->vortex_num_per_row*mesh->vortex_num_per_column*2+mesh->vortex_num_per_row*mesh->vortex_num_per_column/4;
    //int indices[is_size]; 
    //for (int n=0; n<is_size; n++){
    //	indices[n]=n;
    //}   

    //ISCreateGeneral(comm, is_size, indices, PETSC_COPY_VALUES, &isrowcol);

    MatDuplicate(mesh->A, MAT_DO_NOT_COPY_VALUES, &LHS);
    //MatCreateSubMatrix(LHS, isrowcol, isrowcol, MAT_INITIAL_MATRIX, &LHS_sub);
    MatDestroy(&Workspace);
};

//Iterative solver
void NSE2dSolver::SolverSetup(){
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

void NSE2dSolver::ForwardStep(){
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

void NSE2dSolver::solve(bool flag)
{
	VecZeroEntries(X);
	time = 0.0;

	for (int i = 0; i < timeSteps; ++i){
	    //std::cout << "#################" << " level " << level << ", step " << i+1 << " #################" << std::endl;
	    //std::clock_t c_start = std::clock();
            //auto wcts = std::chrono::system_clock::now();	

	    time = time+deltaT;
	    ForwardStep();

	    if (abs(time-0.5) <1e-6){
                VecCopy(X, X_snap);
                solve4QoI();
	        if (flag == 1){
		    return;
		}
	    }	
            //std::clock_t c_end = std::clock();
            //double time_elapsed_ms = (c_end-c_start)/ (double)CLOCKS_PER_SEC;
            //std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
            //std::cout << "wall time " << wctduration.count() << " cpu  time: " << time_elapsed_ms << std::endl;	
	}
	solve4Obs();
};

void NSE2dSolver::priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag)
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

double NSE2dSolver::solve4QoI(){
    return QoiOutput();
};

double NSE2dSolver::solve4Obs(){
    return ObsOutput();
};

double NSE2dSolver::ObsOutput(){
    double obs;
    VecDot(intVecObs, X, &obs);
    obs = 100.*obs;
    return obs;	
};

double NSE2dSolver::QoiOutput(){
    double qoi;
    VecDot(intVecQoi, X_snap, &qoi);
    qoi = 100.*qoi;
    return qoi;
};

double NSE2dSolver::lnLikelihood(){
	double obsResult = ObsOutput();
	double lnLikelihood = -0.5/noiseVariance*pow(obsResult-obs,2);
	return lnLikelihood;
};
