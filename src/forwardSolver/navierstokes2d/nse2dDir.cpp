#include "nse2dDir.h"
#include "nseforcing.h"

PetscErrorCode MyKSPMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *ctx)
{
	PetscPrintf(PETSC_COMM_SELF, "iteration %D KSP Residual norm %14.12e \n", n, rnorm);
	return 0;
}

NSE2dDirSolver::NSE2dDirSolver(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_, DMBoundaryType btype_):comm(comm_), level(level_), num_term(num_term_), noiseVariance(noiseVariance_){
    mesh      = new structureMesh2D(comm_, level_, 3, Q1, btype_);
    obs       = -0.91666;
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

void NSE2dDirSolver::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};


void NSE2dDirSolver::LinearSystemSetup()
{
    AssembleM(mesh->M, mesh->meshDM);
    AssembleA(mesh->A, mesh->meshDM, nu);
    AssembleD(mesh->D, mesh->meshDM);
    AssembleG(mesh->G, mesh->meshDM);
    AssembleQ(mesh->Q, mesh->meshDM);

    Mat Workspace;
    MatScale(mesh->M, 1.0/deltaT);
    MatAXPY(mesh->G,1.0,mesh->Q, DIFFERENT_NONZERO_PATTERN);
    MatPtAP(mesh->G,mesh->D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Workspace);
    MatAXPY(mesh->A,1.0,mesh->M, SAME_NONZERO_PATTERN);
    MatAXPY(mesh->A,1.0,Workspace, DIFFERENT_NONZERO_PATTERN);

    switch (mesh->btype)
    {
        case DM_BOUNDARY_PERIODIC:
        {
            DMCreateMatrix(mesh->meshDM, &LHS);
            MatDiagonalSet(LHS,X,INSERT_VALUES);
            MatAssemblyBegin(LHS,MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(LHS,MAT_FINAL_ASSEMBLY);
            break;
        }

        case DM_BOUNDARY_NONE:
        {
            MatCreate(PETSC_COMM_SELF, &LHS);
            MatSetSizes(LHS, PETSC_DECIDE, PETSC_DECIDE, 3*pow(mesh->vortex_num_per_row+1,2), 3*pow(mesh->vortex_num_per_row+1,2));
            MatSetType(LHS, MATAIJ);
            MatSeqAIJSetPreallocation(LHS, 12, NULL);
            MatMPIAIJSetPreallocation(LHS, 12, NULL, 12, NULL);
            MatDiagonalSet(LHS,X,INSERT_VALUES);
            MatAssemblyBegin(LHS,MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(LHS,MAT_FINAL_ASSEMBLY); 
            break;
        }

        default:
        {
            break;
        }
    }

    MatAXPY(LHS,1.0,mesh->A,DIFFERENT_NONZERO_PATTERN);
    MatSetOption(LHS,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);
    // AttachNullspace(mesh->meshDM,LHS);
    MatDestroy(&Workspace);
}

//Iterative solver
void NSE2dDirSolver::SolverSetup(){
    LinearSystemSetup();

	KSPCreate(comm, &ksp);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCFIELDSPLIT);
	PCFieldSplitSetDetectSaddlePoint(pc, PETSC_TRUE);
	PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, NULL);
	// KSPMonitorSet(ksp, MyKSPMonitor, NULL, 0);
	KSPSetTolerances(ksp, 1e-8, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
	KSPSetFromOptions(ksp);
	KSPSetPC(ksp, pc);
}

void NSE2dDirSolver::AttachNullspace(DM dmSol,Mat A)
{
    double        nrm;
    Vec           NullSpaceVec;
    MatNullSpace  Sp;
    DMCreateGlobalVector(dmSol,&NullSpaceVec);
    for (int i=0; i<(mesh->vortex_num_per_column+1)*(mesh->vortex_num_per_column+1);++i){
        VecSetValue(NullSpaceVec, 3*i+2, 1.0, INSERT_VALUES);
    }
    VecAssemblyBegin(NullSpaceVec);
    VecAssemblyEnd(NullSpaceVec);
    VecNorm(NullSpaceVec,NORM_2,&nrm);
    VecScale(NullSpaceVec,1.0/nrm);

    MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_FALSE,1,&NullSpaceVec,&Sp);

    VecDestroy(&NullSpaceVec);
    MatSetNullSpace(A,Sp);
    MatNullSpaceDestroy(&Sp);
}

void NSE2dDirSolver::ForwardStep(){
    MatZeroEntries(mesh->C);
    VecZeroEntries(mesh->f);

    AssembleC(mesh->C,mesh->meshDM,X,mesh->l2gmapping);

	MatCopy(mesh->A,LHS,SUBSET_NONZERO_PATTERN);
    MatAXPY(LHS,1.0,mesh->C,SUBSET_NONZERO_PATTERN);

    AssembleF(mesh->f,mesh->meshDM,time,forcing,samples.get(),num_term);
    MatMultAdd(mesh->M, X, mesh->f, RHS);

    switch (mesh->btype)
    {
        case DM_BOUNDARY_PERIODIC:
        {
            break;
        }        

        case DM_BOUNDARY_NONE:{
            // Vec solution;
            // VecDuplicate(X, &solution);
            // VecSet(solution, 0.0);

            ApplyBoundaryCondition(LHS,NULL,RHS,mesh->bottomUx);
            ApplyBoundaryCondition(LHS,NULL,RHS,mesh->bottomUy);
            ApplyBoundaryCondition(LHS,NULL,RHS,mesh->leftUx);
            ApplyBoundaryCondition(LHS,NULL,RHS,mesh->leftUy);
            ApplyBoundaryCondition(LHS,NULL,RHS,mesh->rightUx);
            ApplyBoundaryCondition(LHS,NULL,RHS,mesh->rightUy);
            ApplyBoundaryCondition(LHS,NULL,RHS,mesh->topUx);
            ApplyBoundaryCondition(LHS,NULL,RHS,mesh->topUy);

            // VecDestroy(&solution);

            break;
        }

        default:
        {
            break;
        }
    }

	KSPSetOperators(ksp,LHS,LHS);
	KSPSolve(ksp, RHS, X);

    VecISSet(X,mesh->bottomUx,0.0);
    VecISSet(X,mesh->bottomUy,0.0);
    VecISSet(X,mesh->leftUx,0.0);
    VecISSet(X,mesh->leftUy,0.0);
    VecISSet(X,mesh->rightUx,0.0);
    VecISSet(X,mesh->rightUy,0.0);
    VecISSet(X,mesh->topUx,0.0);
    VecISSet(X,mesh->topUy,0.0);

    // VecView(X, PETSC_VIEWER_STDOUT_WORLD);
};

void NSE2dDirSolver::solve(bool flag)
{
	VecZeroEntries(X);
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

void NSE2dDirSolver::priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag)
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

double NSE2dDirSolver::solve4QoI(){
    VecZeroEntries(intVecQoi); 
    AssembleIntegralOperator2(intVecQoi,mesh->meshDM,0.5);
    return QoiOutput();
};

double NSE2dDirSolver::solve4Obs(){
    VecZeroEntries(intVecObs);
    AssembleIntegralOperator2(intVecObs,mesh->meshDM,1.5);
    return ObsOutput();
};

double NSE2dDirSolver::ObsOutput(){
    double obs=0;
    VecDot(intVecObs, X, &obs);
    obs = 20.*obs; 
    return obs;	
}

double NSE2dDirSolver::QoiOutput(){
    double qoi=0;
    VecDot(intVecQoi, X_snap, &qoi);
    qoi = 20.*qoi;
    return qoi;
}

double NSE2dDirSolver::lnLikelihood(){
	double obsResult = ObsOutput();
	double lnLikelihood = -0.5/noiseVariance*pow(obsResult-obs,2);
    // std::cout << samples[0] << " " << obsResult << " " << obs << " " << lnLikelihood << std::endl;
	return lnLikelihood;
}


void NSE2dDirSolver::getValues(double x[], double y[], double *ux, double *uy, int size)
{
    int         column,row,column2,row2;
    double      epsilon,eta;
    double      element_h=1.0/mesh->vortex_num_per_row;
    double      coord[8],u[4],v[4];
	double 		gp_xi[2],Ni_p[4];
    VortexDOF   **states;

    DMDAVecGetArray(mesh->meshDM,X,&states);
    for (int i=0;i<size;i++){
        for (int j=0;j<size;j++){
            column = x[j] / element_h;
            row    = y[i] / element_h;
            
            if (column == mesh->vortex_num_per_row){
                    column -= 1;
            }
            if (row == mesh->vortex_num_per_row){
                    row -= 1;
            } 

            coord[0]=element_h*column;
            coord[1]=element_h*column;
            coord[2]=element_h*(column+1);
            coord[3]=element_h*(column+1);
            coord[4]=element_h*row;
            coord[5]=element_h*(row+1);
            coord[6]=element_h*(row+1);
            coord[7]=element_h*row;

            epsilon = (x[j]-coord[0])/element_h*2.0-1.0;
            eta     = (y[i]-coord[4])/element_h*2.0-1.0;

            column2=column+1;row2=row+1;

            switch (mesh->btype)
            {
                case DM_BOUNDARY_PERIODIC:
                {
                    if (column2==mesh->vortex_num_per_row) {column2=0;}
                    if (row2==mesh->vortex_num_per_row) {row2=0;}
                }

                case DM_BOUNDARY_NONE:
                {
                    break;
                }

                default:
                {
                    std::cout << "Warning: Boundary condition not specified" << std::endl;
                }
            }

            u[0]=states[row][column].u;u[1]=states[row2][column].u;u[2]=states[row2][column2].u;u[3]=states[row][column2].u;
            v[0]=states[row][column].v;v[1]=states[row2][column].v;v[2]=states[row2][column2].v;v[3]=states[row][column2].v;

            gp_xi[0]=epsilon;
            gp_xi[1]=eta;

            ShapeFunctionQ12D_Evaluate(gp_xi,Ni_p);

            ux[i*size+j]=u[0]*Ni_p[0]+u[1]*Ni_p[1]+u[2]*Ni_p[2]+u[3]*Ni_p[3];
            uy[i*size+j]=v[0]*Ni_p[0]+v[1]*Ni_p[1]+v[2]*Ni_p[2]+v[3]*Ni_p[3];

            // std::cout << ux[i*size+j] << " " << uy[i*size+j] << std::endl;// " " << column << " " << column2 << " " << u[0] << " " << u[1] << " " << u[2] << " " << u[3] << " " << std::endl;
        }   
    }
    DMDAVecRestoreArray(mesh->meshDM,X,&states);
}


void NSE2dDirSolver::getGradients(double x[], double y[], int size)
{
    int         column,row,column2,row2;
    double      epsilon,eta;
    double      element_h=1.0/mesh->vortex_num_per_row;
    double      coord[8],u[4],v[4],uxx,uxy,uyx,uyy;
	double 		gp_xi[2],Ni_p[4],GNi_p[2][4],GNx_p[2][4];
    double      J_p;
    VortexDOF   **states;

    std::ofstream uxxOuput("uxx");
    std::ofstream uxyOuput("uxy");
    std::ofstream uyxOuput("uyx");
    std::ofstream uyyOuput("uyy");

    DMDAVecGetArray(mesh->meshDM,X,&states);
    for (int i=0;i<size;i++){
        for (int j=0;j<size;j++){
            column = x[j] / element_h;
            row    = y[i] / element_h;
            
            if (column == mesh->vortex_num_per_row){
                    column -= 1;
            }
            if (row == mesh->vortex_num_per_row){
                    row -= 1;
            } 
            coord[0]=element_h*column;
            coord[1]=element_h*row;
            coord[2]=element_h*column;
            coord[3]=element_h*(row+1);
            coord[4]=element_h*(column+1);
            coord[5]=element_h*(row+1);
            coord[6]=element_h*(column+1);
            coord[7]=element_h*row;

            epsilon = (x[j]-coord[0])/element_h*2.0-1.0;
            eta     = (y[i]-coord[1])/element_h*2.0-1.0;

            column2=column+1;row2=row+1;
            
            switch (mesh->btype)
            {
                case DM_BOUNDARY_PERIODIC:
                {
                    if (column2==mesh->vortex_num_per_row) {column2=0;}
                    if (row2==mesh->vortex_num_per_row) {row2=0;}
                }

                case DM_BOUNDARY_NONE:
                {
                    break;
                }

                default:
                {
                    std::cout << "Warning: Boundary condition not specified" << std::endl;
                }
            }

            u[0]=states[row][column].u;u[1]=states[row2][column].u;u[2]=states[row2][column2].u;u[3]=states[row][column2].u;
            v[0]=states[row][column].v;v[1]=states[row2][column].v;v[2]=states[row2][column2].v;v[3]=states[row][column2].v;

            gp_xi[0]=epsilon;
            gp_xi[1]=eta;

            ShapeFunctionQ12D_Evaluate(gp_xi,Ni_p);
            ShapeFunctionQ12D_Evaluate_dxi(gp_xi,GNi_p);
            ShapeFunctionQ12D_Evaluate_dx(GNi_p,GNx_p,coord,&J_p);

            uxx=u[0]*GNx_p[0][0]+u[1]*GNx_p[0][1]+u[2]*GNx_p[0][2]+u[3]*GNx_p[0][3];
            uxy=u[0]*GNx_p[1][0]+u[1]*GNx_p[1][1]+u[2]*GNx_p[1][2]+u[3]*GNx_p[1][3];
            uyx=v[0]*GNx_p[0][0]+v[1]*GNx_p[0][1]+v[2]*GNx_p[0][2]+v[3]*GNx_p[0][3];
            uyy=v[0]*GNx_p[1][0]+v[1]*GNx_p[1][1]+v[2]*GNx_p[1][2]+v[3]*GNx_p[1][3];

            uxxOuput << uxx << ",";
            uxyOuput << uxy << ",";
            uyxOuput << uyx << ",";
            uyyOuput << uyy << ","; 
        }   
        uxxOuput << std::endl;
        uxyOuput << std::endl;
        uyxOuput << std::endl;
        uyyOuput << std::endl;
    }
    DMDAVecRestoreArray(mesh->meshDM,X,&states);
}


void NSE2dDirSolver::getGradients(double x[], double y[], double uxx[], double uxy[], double uyx[], double uyy[], int size)
{
    int         column,row,column2,row2;
    double      epsilon,eta;
    double      element_h=1.0/mesh->vortex_num_per_row;
    double      coord[8],u[4],v[4];
	double 		gp_xi[2],Ni_p[4],GNi_p[2][4],GNx_p[2][4];
    double      J_p;
    VortexDOF   **states;

    DMDAVecGetArray(mesh->meshDM,X,&states);
    for (int i=0;i<size;i++){
        for (int j=0;j<size;j++){
            column = x[j] / element_h;
            row    = y[i] / element_h;
            
            if (column == mesh->vortex_num_per_row){
                    column -= 1;
            }
            if (row == mesh->vortex_num_per_row){
                    row -= 1;
            } 
            coord[0]=element_h*column;
            coord[1]=element_h*row;
            coord[2]=element_h*column;
            coord[3]=element_h*(row+1);
            coord[4]=element_h*(column+1);
            coord[5]=element_h*(row+1);
            coord[6]=element_h*(column+1);
            coord[7]=element_h*row;

            epsilon = (x[j]-coord[0])/element_h*2.0-1.0;
            eta     = (y[i]-coord[1])/element_h*2.0-1.0;

            column2=column+1;row2=row+1;
            
            switch (mesh->btype)
            {
                case DM_BOUNDARY_PERIODIC:
                {
                    if (column2==mesh->vortex_num_per_row) {column2=0;}
                    if (row2==mesh->vortex_num_per_row) {row2=0;}
                }

                case DM_BOUNDARY_NONE:
                {
                    break;
                }

                default:
                {
                    std::cout << "Warning: Boundary condition not specified" << std::endl;
                }
            }

            u[0]=states[row][column].u;u[1]=states[row2][column].u;u[2]=states[row2][column2].u;u[3]=states[row][column2].u;
            v[0]=states[row][column].v;v[1]=states[row2][column].v;v[2]=states[row2][column2].v;v[3]=states[row][column2].v;

            gp_xi[0]=epsilon;
            gp_xi[1]=eta;

            ShapeFunctionQ12D_Evaluate(gp_xi,Ni_p);
            ShapeFunctionQ12D_Evaluate_dxi(gp_xi,GNi_p);
            ShapeFunctionQ12D_Evaluate_dx(GNi_p,GNx_p,coord,&J_p);

            uxx[i*size+j]=u[0]*GNx_p[0][0]+u[1]*GNx_p[0][1]+u[2]*GNx_p[0][2]+u[3]*GNx_p[0][3];
            uxy[i*size+j]=u[0]*GNx_p[1][0]+u[1]*GNx_p[1][1]+u[2]*GNx_p[1][2]+u[3]*GNx_p[1][3];
            uyx[i*size+j]=v[0]*GNx_p[0][0]+v[1]*GNx_p[0][1]+v[2]*GNx_p[0][2]+v[3]*GNx_p[0][3];
            uyy[i*size+j]=v[0]*GNx_p[1][0]+v[1]*GNx_p[1][1]+v[2]*GNx_p[1][2]+v[3]*GNx_p[1][3];

            // std::cout << "Gradients " << x[j] << " " << y[i] << " " << epsilon << " " << eta << " " << u[0] << " " << u[1] << " " << u[2] << " " << u[3] << " " << GNx_p[0][0] << " " << GNx_p[0][1]<< " " << GNx_p[0][2]<< " " << GNx_p[0][3] << " " << uxx[i*size+j] << std::endl;
        }   
    }
    DMDAVecRestoreArray(mesh->meshDM,X,&states);
}
