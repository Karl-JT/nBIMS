#include "nse2dCNDir.h"
#include "nseforcing.h"

PetscErrorCode MyKSPMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *ctx)
{
	PetscPrintf(PETSC_COMM_SELF, "iteration %D KSP Residual norm %14.12e \n", n, rnorm);
	return 0;
}

PetscErrorCode MySNESMonitor(SNES snes, PetscInt n, PetscReal fnorm, void *ctx)
{
	PetscPrintf(PETSC_COMM_WORLD,"iter = %D, SNES Function norm %g\n", n, (double)fnorm);
	return 0;	
}

NSE2dSolverCNDir::NSE2dSolverCNDir(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_, DMBoundaryType btype_):comm(comm_), level(level_), num_term(num_term_), noiseVariance(noiseVariance_){
    mesh      = new structureMesh2D(comm_, level_, 3, Q1, btype_);
    obs       = -0.91666;
    timeSteps = std::pow(2, level_+1);
	deltaT    = tMax/timeSteps;

	samples   = std::make_unique<double[]>(num_term_);

	DMCreateGlobalVector(mesh->meshDM,&Xn);
	DMCreateGlobalVector(mesh->meshDM,&r);
	DMCreateGlobalVector(mesh->meshDM,&intVecObs);
	DMCreateGlobalVector(mesh->meshDM,&intVecQoi);

    VecZeroEntries(Xn);
    VecDuplicate(Xn,&Xn1);
    VecDuplicate(Xn,&X_snap);
    VecDuplicate(Xn,&X_bar);
    VecDuplicate(Xn,&RHS);

    SolverSetup();
};

void NSE2dSolverCNDir::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};

void NSE2dSolverCNDir::LinearSystemSetup()
{
    AssembleM(mesh->M, mesh->meshDM);
    MatScale(mesh->M, 1.0/deltaT);

    AssembleA(mesh->A, mesh->meshDM, nu);
    AssembleG(mesh->G, mesh->meshDM);
    AssembleQ(mesh->Q, mesh->meshDM);
    AssembleD(mesh->D, mesh->meshDM);

    Mat Workspace;
    MatDuplicate(mesh->M, MAT_COPY_VALUES, &Workspace);
    MatAXPY(Workspace,1.0,mesh->Q,DIFFERENT_NONZERO_PATTERN);
    MatAXPY(Workspace,1.0,mesh->G,DIFFERENT_NONZERO_PATTERN);
    MatPtAP(Workspace,mesh->D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&LHS);  

    AssembleJ(mesh->J,mesh->meshDM,Xn,mesh->l2gmapping);
    // MatAXPY(mesh->J,1.0,LHS,DIFFERENT_NONZERO_PATTERN);

    switch (mesh->btype)
    {
        case DM_BOUNDARY_PERIODIC:
        {
            MatDuplicate(mesh->J,MAT_DO_NOT_COPY_VALUES,&J);
            break;
        }

        case DM_BOUNDARY_NONE:
        {
            MatCreate(PETSC_COMM_SELF, &J);
            MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 3*pow(mesh->vortex_num_per_row+1,2), 3*pow(mesh->vortex_num_per_row+1,2));
            MatSetType(J, MATAIJ);
            MatSeqAIJSetPreallocation(J, 12, NULL);
            MatMPIAIJSetPreallocation(J, 12, NULL, 12, NULL);
            MatDiagonalSet(J,Xn,INSERT_VALUES);
            MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); 

            MatAXPY(J,1.0,mesh->J,DIFFERENT_NONZERO_PATTERN);
            break;
        }

        default:
        {
            break;
        }
    }
    MatAXPY(J,1.0,LHS,DIFFERENT_NONZERO_PATTERN);
    MatDuplicate(J,MAT_DO_NOT_COPY_VALUES,&Jt);
    MatSetOption(Jt,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);

    MatDuplicate(LHS,MAT_COPY_VALUES,&LHSpA);
    MatAXPY(LHSpA,0.5,mesh->A,DIFFERENT_NONZERO_PATTERN);

    MatDestroy(&Workspace);
}

//Iterative solver
void NSE2dSolverCNDir::SolverSetup(){
    LinearSystemSetup();

    SNESCreate(comm, &snes);
	SNESSetFunction(snes, r, FormFunctionStatic, this);
	SNESSetJacobian(snes, J, J, FormJacobianStatic, this);
	// SNESSetJacobian(snes, J, J, SNESComputeJacobianDefault, this);	
	SNESMonitorSet(snes, MySNESMonitor, NULL, NULL);    

	SNESGetKSP(snes, &ksp);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCFIELDSPLIT);
	PCFieldSplitSetDetectSaddlePoint(pc, PETSC_TRUE);
	PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, NULL);
	// KSPMonitorSet(ksp, MyKSPMonitor, NULL, 0);
	KSPSetTolerances(ksp, 1e-8, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
	KSPSetFromOptions(ksp);
	KSPSetPC(ksp, pc);
	SNESSetFromOptions(snes);
}

void NSE2dSolverCNDir::FormFunction(SNES snes, Vec xx, Vec ff)
{
    VecWAXPY(X_bar,1,Xn,xx);
    VecScale(X_bar,0.5);

    MatZeroEntries(mesh->C);
    AssembleC(mesh->C,mesh->meshDM,X_bar,mesh->l2gmapping);

    Vec ft;
    VecDuplicate(Xn, &ft);
    MatMult(LHS,xx,ft);
    MatMultAdd(mesh->C,X_bar,ft,ft);
    MatMultAdd(mesh->A,X_bar,ft,ft);

    VecISSet(ft,mesh->bottomUx,0.0);
    VecISSet(ft,mesh->bottomUy,0.0);
    VecISSet(ft,mesh->leftUx,0.0);
    VecISSet(ft,mesh->leftUy,0.0);
    VecISSet(ft,mesh->rightUx,0.0);
    VecISSet(ft,mesh->rightUy,0.0);
    VecISSet(ft,mesh->topUx,0.0);
    VecISSet(ft,mesh->topUy,0.0);

    VecCopy(ft,ff);
    VecDestroy(&ft);
}

void NSE2dSolverCNDir::FormJacobian(SNES snes, Vec xx, Mat jac, Mat B)
{
    VecWAXPY(X_bar,1,Xn,xx);
    VecScale(X_bar,0.5);

    MatZeroEntries(mesh->J);
    AssembleJ(mesh->J,mesh->meshDM,X_bar,mesh->l2gmapping);

    MatCopy(LHSpA,Jt,SUBSET_NONZERO_PATTERN);  
    MatAXPY(Jt,0.5,mesh->J,SUBSET_NONZERO_PATTERN);

    switch (mesh->btype)
    {
        case DM_BOUNDARY_PERIODIC:
        {
            break;
        }        

        case DM_BOUNDARY_NONE:{
            ApplyBoundaryCondition(Jt,NULL,RHS,mesh->bottomUx);
            ApplyBoundaryCondition(Jt,NULL,RHS,mesh->bottomUy);
            ApplyBoundaryCondition(Jt,NULL,RHS,mesh->leftUx);
            ApplyBoundaryCondition(Jt,NULL,RHS,mesh->leftUy);
            ApplyBoundaryCondition(Jt,NULL,RHS,mesh->rightUx);
            ApplyBoundaryCondition(Jt,NULL,RHS,mesh->rightUy);
            ApplyBoundaryCondition(Jt,NULL,RHS,mesh->topUx);
            ApplyBoundaryCondition(Jt,NULL,RHS,mesh->topUy);
            break;      
        }

        default:
        {
            break;
        }
    }

    MatCopy(Jt, jac, SAME_NONZERO_PATTERN);
}

void NSE2dSolverCNDir::ForwardStep(){
    VecZeroEntries(mesh->f);
    AssembleF(mesh->f,mesh->meshDM,time-0.5*deltaT,forcing,samples.get(),num_term);
    MatMultAdd(mesh->M, Xn, mesh->f, RHS);

    // std::cout << "RHS" << std::endl;
    // VecView(RHS,PETSC_VIEWER_STDOUT_WORLD);
    VecISSet(RHS,mesh->bottomUx,0.0);
    VecISSet(RHS,mesh->bottomUy,0.0);
    VecISSet(RHS,mesh->leftUx,0.0);
    VecISSet(RHS,mesh->leftUy,0.0);
    VecISSet(RHS,mesh->rightUx,0.0);
    VecISSet(RHS,mesh->rightUy,0.0);
    VecISSet(RHS,mesh->topUx,0.0);
    VecISSet(RHS,mesh->topUy,0.0);

    SNESSolve(snes, RHS, Xn1);

    VecISSet(Xn1,mesh->bottomUx,0.0);
    VecISSet(Xn1,mesh->bottomUy,0.0);
    VecISSet(Xn1,mesh->leftUx,0.0);
    VecISSet(Xn1,mesh->leftUy,0.0);
    VecISSet(Xn1,mesh->rightUx,0.0);
    VecISSet(Xn1,mesh->rightUy,0.0);
    VecISSet(Xn1,mesh->topUx,0.0);
    VecISSet(Xn1,mesh->topUy,0.0);

    VecCopy(Xn1, Xn);
    // std::cout << "solution" << std::endl;
    // VecView(Xn1,PETSC_VIEWER_STDOUT_WORLD);
};

void NSE2dSolverCNDir::solve(bool flag)
{
	VecZeroEntries(Xn);
    VecZeroEntries(Xn1);
    VecZeroEntries(X_snap);
	time = 0.0;

	for (int i = 0; i < timeSteps; ++i){
		// std::cout << "#################" << " level " << level << ", step " << i+1 << " #################" << std::endl;
		// std::clock_t c_start = std::clock();
		// auto wcts = std::chrono::system_clock::now();		
		time = time+deltaT;
		ForwardStep();

		if (abs(time-0.5) <1e-6){
            VecCopy(Xn1, X_snap);
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

void NSE2dSolverCNDir::priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag)
{
	switch(flag){
        case UNIFORM:
            initialSamples[0] = uniformDistribution(generator);
            break;
        case GAUSSIAN:
            initialSamples[0] = normalDistribution(generator);
            break;
        default:
            initialSamples[0] = normalDistribution(generator);
    }
}

double NSE2dSolverCNDir::solve4QoI(){
    VecZeroEntries(intVecQoi);
    AssembleIntegralOperator2(intVecQoi,mesh->meshDM,0.5);
    return QoiOutput();
};

double NSE2dSolverCNDir::solve4Obs(){
    VecZeroEntries(intVecObs);
    AssembleIntegralOperator2(intVecObs,mesh->meshDM,1.5);
    return ObsOutput();
};

double NSE2dSolverCNDir::ObsOutput(){
    double obs=0;
    VecDot(intVecObs, Xn1, &obs);
    obs = 20.*obs;
    return obs;	
}

double NSE2dSolverCNDir::QoiOutput(){
    double qoi=0;
    VecDot(intVecQoi, X_snap, &qoi);
    qoi = 20.*qoi;
    return qoi;
}

double NSE2dSolverCNDir::lnLikelihood(){
	double obsResult = ObsOutput();
	double lnLikelihood = -0.5/noiseVariance*pow(obsResult-obs,2);
	return lnLikelihood;
}

void NSE2dSolverCNDir::getValues(double x[], double y[], double *ux, double *uy, int size)
{
    int         column,row,column2,row2;
    double      epsilon,eta;
    double      element_h=1.0/mesh->vortex_num_per_row;
    double      coord[8],u[4],v[4];
	double 		gp_xi[2],Ni_p[4];
    VortexDOF   **states;

    DMDAVecGetArray(mesh->meshDM,Xn,&states);
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
            if (column2==mesh->vortex_num_per_row) {column2=0;}
            if (row2==mesh->vortex_num_per_row) {row2=0;}

            u[0]=states[row][column].u;u[1]=states[row2][column].u;u[2]=states[row2][column2].u;u[3]=states[row][column2].u;
            v[0]=states[row][column].v;v[1]=states[row2][column].v;v[2]=states[row2][column2].v;v[3]=states[row][column2].v;

            gp_xi[0]=epsilon;
            gp_xi[1]=eta;

            ShapeFunctionQ12D_Evaluate(gp_xi,Ni_p);

            ux[i*size+j]=u[0]*Ni_p[0]+u[1]*Ni_p[1]+u[2]*Ni_p[2]+u[3]*Ni_p[3];
            uy[i*size+j]=v[0]*Ni_p[0]+v[1]*Ni_p[1]+v[2]*Ni_p[2]+v[3]*Ni_p[3];
        }   
    }
    DMDAVecRestoreArray(mesh->meshDM,Xn,&states);
}


void NSE2dSolverCNDir::getGradients(double x[], double y[], int size)
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

    DMDAVecGetArray(mesh->meshDM,Xn,&states);
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
            if (column2==mesh->vortex_num_per_row) {column2=0;}
            if (row2==mesh->vortex_num_per_row) {row2=0;}

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
    DMDAVecRestoreArray(mesh->meshDM,Xn,&states);
}

void NSE2dSolverCNDir::getGradients(double x[], double y[], double uxx[], double uxy[], double uyx[], double uyy[], int size)
{
    int         column,row,column2,row2;
    double      epsilon,eta;
    double      element_h=1.0/mesh->vortex_num_per_row;
    double      coord[8],u[4],v[4];
	double 		gp_xi[2],Ni_p[4],GNi_p[2][4],GNx_p[2][4];
    double      J_p;
    VortexDOF   **states;

    DMDAVecGetArray(mesh->meshDM,Xn,&states);
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
    DMDAVecRestoreArray(mesh->meshDM,Xn,&states);
}