#include <FEModule.h>

/* //Macro for zero boudnary condition
// #define _ZERO_ROWCOL_i(A,i) {                   
//     PetscInt    KK;                             
//     PetscScalar tmp = A[24*(i)+(i)];            
//     for (KK=0;KK<24;KK++) A[24*(i)+KK]=0.0;     
//     for (KK=0;KK<24;KK++) A[24*KK+(i)]=0.0;     
//     A[24*(i)+(i)] = tmp;}                       

// #define _ZERO_ROW_i(A,i) {                      
//     PetscInt KK;                                
//     for (KK=0;KK<8;KK++) A[8*(i)+KK]=0.0;}

// #define _ZERO_COL_i(A,i) {                      
//     PetscInt KK;                                
//     for (KK=0;KK<8;KK++) A[24*KK+(i)]=0.0;} */

structureMesh2D::structureMesh2D(MPI_Comm comm_, int level_, int dof_, ELEMENT_TYPE etype_, DMBoundaryType btype_): comm(comm_), level(level_), dof(dof_), etype(etype_), btype(btype_)
{
    vortex_num_per_row    = pow(2, level+2);
    vortex_num_per_column = pow(2, level+2);

    switch (btype)
    {
        case DM_BOUNDARY_PERIODIC:
        {
            DMDACreate2d(comm, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, vortex_num_per_column, vortex_num_per_row, PETSC_DECIDE, PETSC_DECIDE, 3, 1, NULL, NULL, &meshDM);
            break;
        }
        
        case DM_BOUNDARY_NONE:
        {
            std::cout << "Generating Dirichelet Boundary Mesh" << std::endl;
            DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, vortex_num_per_column+1, vortex_num_per_row+1, PETSC_DECIDE, PETSC_DECIDE, 3, 1, NULL, NULL, &meshDM);
            ISCreateStride(comm,vortex_num_per_column,0,3,&bottomUx);        
            ISCreateStride(comm,vortex_num_per_column,1,3,&bottomUy);        
            ISCreateStride(comm,vortex_num_per_column,3*(vortex_num_per_column+1),  3*(vortex_num_per_column+1),&leftUx);        
            ISCreateStride(comm,vortex_num_per_column,3*(vortex_num_per_column+1)+1,3*(vortex_num_per_column+1),&leftUy);        
            ISCreateStride(comm,vortex_num_per_column,3*(vortex_num_per_column),    3*(vortex_num_per_column+1),&rightUx);        
            ISCreateStride(comm,vortex_num_per_column,3*(vortex_num_per_column)+1,  3*(vortex_num_per_column+1),&rightUy);        
            ISCreateStride(comm,vortex_num_per_column,3*(vortex_num_per_column+1)*vortex_num_per_column+3,3,&topUx);        
            ISCreateStride(comm,vortex_num_per_column,3*(vortex_num_per_column+1)*vortex_num_per_column+4,3,&topUy);   
            break;     
        }

        default:
        {
            DMDACreate2d(comm, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, vortex_num_per_column, vortex_num_per_row, PETSC_DECIDE, PETSC_DECIDE, 3, 1, NULL, NULL, &meshDM);
            break;
        }
    }

	DMSetMatType(meshDM, MATAIJ);
	// DMSetFromOptions(meshDM);
	DMSetUp(meshDM);
    DMDASetFieldName(meshDM, 0, "u");
    DMDASetFieldName(meshDM, 1, "v");
    DMDASetFieldName(meshDM, 2, "p");
    
    DMDASetUniformCoordinates(meshDM, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    DMGetLocalToGlobalMapping(meshDM,&l2gmapping);

    DMSetMatrixPreallocateOnly(meshDM, PETSC_TRUE);

    switch (btype)
    {
        case DM_BOUNDARY_PERIODIC:
        {
            DMCreateMatrix(meshDM, &A);
            DMCreateMatrix(meshDM, &M);
            DMCreateMatrix(meshDM, &G);	
            DMCreateMatrix(meshDM, &Q);
            DMCreateMatrix(meshDM, &C);
            DMCreateMatrix(meshDM, &J);
            DMCreateMatrix(meshDM, &D);
            DMCreateMatrix(meshDM, &P);
            break;
        }
        
        case DM_BOUNDARY_GHOSTED:
        {
            DMCreateMatrix(meshDM, &A);
            DMCreateMatrix(meshDM, &M);
            DMCreateMatrix(meshDM, &G);	
            DMCreateMatrix(meshDM, &Q);
            DMCreateMatrix(meshDM, &C);
            DMCreateMatrix(meshDM, &J);
            DMCreateMatrix(meshDM, &D);
            DMCreateMatrix(meshDM, &P);
            break;            
        }

        case DM_BOUNDARY_NONE:
        {
            MatCreate(PETSC_COMM_SELF, &M);
            MatCreate(PETSC_COMM_SELF, &A);
            MatCreate(PETSC_COMM_SELF, &G);
            MatCreate(PETSC_COMM_SELF, &Q);
            MatCreate(PETSC_COMM_SELF, &D);
            MatCreate(PETSC_COMM_SELF, &C);
            MatCreate(PETSC_COMM_SELF, &J);
            MatCreate(PETSC_COMM_SELF, &P);

            MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, 3*pow(vortex_num_per_row+1,2), 3*pow(vortex_num_per_row+1,2));
            MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 3*pow(vortex_num_per_row+1,2), 3*pow(vortex_num_per_row+1,2));
            MatSetSizes(G, PETSC_DECIDE, PETSC_DECIDE, 3*pow(vortex_num_per_row+1,2), 3*pow(vortex_num_per_row+1,2));
            MatSetSizes(Q, PETSC_DECIDE, PETSC_DECIDE, 3*pow(vortex_num_per_row+1,2), 3*pow(vortex_num_per_row+1,2));
            MatSetSizes(D, PETSC_DECIDE, PETSC_DECIDE, 3*pow(vortex_num_per_row+1,2), 3*pow(vortex_num_per_row+1,2));
            MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, 3*pow(vortex_num_per_row+1,2), 3*pow(vortex_num_per_row+1,2));
            MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 3*pow(vortex_num_per_row+1,2), 3*pow(vortex_num_per_row+1,2));
            MatSetSizes(P, PETSC_DECIDE, PETSC_DECIDE, 3*pow(vortex_num_per_row+1,2), 3*pow(vortex_num_per_row+1,2));

            MatSetType(M, MATAIJ);
            MatSetType(A, MATAIJ);
            MatSetType(G, MATAIJ);
            MatSetType(Q, MATAIJ);
            MatSetType(D, MATAIJ);
            MatSetType(C, MATAIJ);
            MatSetType(J, MATAIJ);
            MatSetType(P, MATAIJ);

            MatSeqAIJSetPreallocation(M, 12, NULL);
            MatMPIAIJSetPreallocation(M, 12, NULL, 12, NULL);
            MatSeqAIJSetPreallocation(P, 12, NULL);
            MatMPIAIJSetPreallocation(P, 12, NULL, 12, NULL);
            MatSeqAIJSetPreallocation(A, 12, NULL);
            MatMPIAIJSetPreallocation(A, 12, NULL, 12, NULL);
            MatSeqAIJSetPreallocation(G, 12, NULL);
            MatMPIAIJSetPreallocation(G, 12, NULL, 12, NULL);            
            MatSeqAIJSetPreallocation(Q, 24, NULL);
            MatMPIAIJSetPreallocation(Q, 24, NULL, 24, NULL);            
            MatSeqAIJSetPreallocation(D, 24, NULL);
            MatMPIAIJSetPreallocation(D, 24, NULL, 24, NULL);
            MatSeqAIJSetPreallocation(C, 12, NULL);
            MatMPIAIJSetPreallocation(C, 12, NULL, 12, NULL);            
            MatSeqAIJSetPreallocation(J, 24, NULL);
            MatMPIAIJSetPreallocation(J, 24, NULL, 24, NULL);

            MatSetLocalToGlobalMapping(M,l2gmapping,l2gmapping);
            MatSetLocalToGlobalMapping(P,l2gmapping,l2gmapping);
            MatSetLocalToGlobalMapping(A,l2gmapping,l2gmapping);
            MatSetLocalToGlobalMapping(G,l2gmapping,l2gmapping);
            MatSetLocalToGlobalMapping(Q,l2gmapping,l2gmapping);
            MatSetLocalToGlobalMapping(D,l2gmapping,l2gmapping);
            MatSetLocalToGlobalMapping(C,l2gmapping,l2gmapping);
            MatSetLocalToGlobalMapping(J,l2gmapping,l2gmapping);

            int starts[2],dims[3];
            DMDAGetGhostCorners(meshDM,&starts[0],&starts[1],NULL,&dims[0],&dims[1],NULL);

            MatSetStencil(M,2,dims,starts,3);
            MatSetStencil(P,2,dims,starts,3);
            MatSetStencil(A,2,dims,starts,3);
            MatSetStencil(G,2,dims,starts,3);
            MatSetStencil(Q,2,dims,starts,3);
            MatSetStencil(D,2,dims,starts,3);
            MatSetStencil(C,2,dims,starts,3);
            MatSetStencil(J,2,dims,starts,3);

            break;     
        }

        default:
        {
            break;
        }
    }

    //int n_size = vortex_num_per_row*vortex_num_per_column*3;
    //for(int n=0; n < n_size; n++){
    //    MatSetValue(A, n, n, 0, INSERT_VALUES);
    //}

    DMCreateGlobalVector(meshDM, &f);
    VecZeroEntries(f);
};

void GetElementCoordinates2D(DMDACoor2d **coords, int i, int j, double el_coords[])
{
    if (i==-1){
        // std::cout << "access ghost point" << std::endl;
        el_coords[0] = coords[j][i].x;
        el_coords[2] = coords[j+1][i].x;
        el_coords[4] = 1.0;
        el_coords[6] = 1.0;        
    } else {
        el_coords[0] = coords[j][i].x;
        el_coords[2] = coords[j+1][i].x;
        el_coords[4] = coords[j+1][i+1].x;
        el_coords[6] = coords[j][i+1].x;
    }

    if (j==-1){
        // std::cout << "access ghost point" << std::endl;
        el_coords[1] = coords[j][i].y;
        el_coords[3] = 1.0;
        el_coords[5] = 1.0;
        el_coords[7] = coords[j][i+1].y;      
    } else {
        el_coords[1] = coords[j][i].y;
        el_coords[3] = coords[j+1][i].y;
        el_coords[5] = coords[j+1][i+1].y;
        el_coords[7] = coords[j][i+1].y;
    }
};

void ConstructGaussQuadrature2D(int ngp, double gp_xi[][2], double gp_weight[])
{
    double quadx[ngp];
    double quadw[ngp];
    gauleg(ngp, quadx, quadw);  //gaussian legendre

    for (int i=0; i<ngp; ++i){
        for (int j=0; j<ngp; ++j){
            gp_xi[ngp*i+j][0]=quadx[i];
            gp_xi[ngp*i+j][1]=quadx[j];
            gp_weight[ngp*i+j]=quadw[i]*quadw[j];
        }
    }
};

void ShapeFunctionQ12D_Evaluate(double xi_p[], double Ni_p[])
{
	double xi  = xi_p[0];
	double eta = xi_p[1];

	Ni_p[0] = 0.25*(1.0-xi)*(1.0-eta);
	Ni_p[1] = 0.25*(1.0-xi)*(1.0+eta);
	Ni_p[2] = 0.25*(1.0+xi)*(1.0+eta);
	Ni_p[3] = 0.25*(1.0+xi)*(1.0-eta); 
};

void ShapeFunctionQ12D_Evaluate_dxi(double xi_p[], double GNi[][4])
{
	double xi  = xi_p[0];
	double eta = xi_p[1];

	GNi[0][0] = -0.25+0.25*eta;	
	GNi[0][1] = -0.25-0.25*eta;
	GNi[0][2] =  0.25+0.25*eta;
	GNi[0][3] =  0.25-0.25*eta;

	GNi[1][0] = -0.25+0.25*xi;	
	GNi[1][1] =  0.25-0.25*xi;
	GNi[1][2] =  0.25+0.25*xi;
	GNi[1][3] = -0.25-0.25*xi;
}

void ShapeFunctionQ12D_Evaluate_dx(double GNi[][4], double GNx[][4], double coords[], double *det_J)
{
	double  J00,J01,J10,J11;
	double  iJ[2][2];
    J00=0;J01=0;J10=0;J11=0;

	for (int n = 0; n < 4; ++n){
		double cx = coords[2*n];
		double cy = coords[2*n+1];

		J00=J00+GNi[0][n]*cx;
		J01=J01+GNi[0][n]*cy;
		J10=J10+GNi[1][n]*cx;
		J11=J11+GNi[1][n]*cy;
	}

	*det_J = J00*J11 - J01*J10;

	iJ[0][0] = J11/ *det_J;
	iJ[0][1] =-J01/ *det_J;
	iJ[1][0] =-J10/ *det_J;
	iJ[1][1] = J00/ *det_J;

	for (int n=0; n< 4; ++n){
		GNx[0][n] = GNi[0][n]*iJ[0][0]+GNi[1][n]*iJ[0][1];
		GNx[1][n] = GNi[0][n]*iJ[1][0]+GNi[1][n]*iJ[1][1];
	}
}

void FormStressOperatorQ12D(double Kex[],double Key[], double coords[], double eta[])
{
	double 				gp_xi[4][2];
	double 				gp_weight[4];
	double 				GNi_p[2][4], GNx_p[2][4];
	double 				J_p;
	double 				d_dx_i, d_dy_i;
    double              B[3][8];
    double              tildeD[3];

	ConstructGaussQuadrature2D(2, gp_xi, gp_weight);

	for (int p=0; p <4; ++p){
		ShapeFunctionQ12D_Evaluate_dxi(gp_xi[p], GNi_p);
		ShapeFunctionQ12D_Evaluate_dx(GNi_p, GNx_p, coords, &J_p);

		for (int i = 0; i < 4; ++i){
			d_dx_i = GNx_p[0][i];
			d_dy_i = GNx_p[1][i];

            B[0][2*i] = d_dx_i;B[0][2*i+1]=0;
            B[1][2*i] = 0;B[1][2*i+1]=d_dy_i;
            B[2][2*i] = d_dy_i;B[2][2*i+1]=d_dx_i;
		}

        tildeD[0] = 1.0*gp_weight[p]*J_p*eta[p];
        tildeD[1] = 1.0*gp_weight[p]*J_p*eta[p];

        tildeD[2] = 1.0*gp_weight[p]*J_p*eta[p];

        for (int i=0; i<8; i+=2){
            for (int j=0; j<8; j+=2){
                for (int k=0; k<3; ++k){
                    Kex[i/2+4*j/2] += B[k][i]*tildeD[k]*B[k][j];
                }
            }
        }
        for (int i=1; i<8; i+=2){
            for (int j=1; j<8; j+=2){
                for (int k=0; k<3; ++k){
                    Key[(i-1)/2+4*(j-1)/2] += B[k][i]*tildeD[k]*B[k][j];
                }
            }
        }
	}
}


void FormStressOperatorQ12Dnu(double Kex[],double Key[], double coords[], double(*Visc)(double, double, double[], int), double samples[], int sampleSize)
{
	double 				gp_xi[4][2];
	double 				gp_weight[4];
	double 				GNi_p[2][4], GNx_p[2][4];
    double              Ni_p[4];
	double 				J_p;
	double 				d_dx_i, d_dy_i;
    double              B[3][8];
    double              tildeD[3];
    double              xCoord,yCoord,nu;

	ConstructGaussQuadrature2D(2, gp_xi, gp_weight);

	for (int p=0; p <4; ++p){
        ShapeFunctionQ12D_Evaluate(gp_xi[p],Ni_p);
		ShapeFunctionQ12D_Evaluate_dxi(gp_xi[p], GNi_p);
		ShapeFunctionQ12D_Evaluate_dx(GNi_p, GNx_p, coords, &J_p);

        xCoord=0;yCoord=0;
        for (int i=0; i<4; i++) {
            xCoord += Ni_p[i]*coords[i*2];
            yCoord += Ni_p[i]*coords[i*2+1];
        }
        nu = Visc(xCoord,yCoord,samples,sampleSize);

		for (int i = 0; i < 4; ++i){
			d_dx_i = GNx_p[0][i];
			d_dy_i = GNx_p[1][i];

            B[0][2*i] = d_dx_i;B[0][2*i+1]=0;
            B[1][2*i] = 0;B[1][2*i+1]=d_dy_i;
            B[2][2*i] = d_dy_i;B[2][2*i+1]=d_dx_i;
		}

        tildeD[0] = 1.0*gp_weight[p]*J_p*nu;
        tildeD[1] = 1.0*gp_weight[p]*J_p*nu;

        tildeD[2] = 1.0*gp_weight[p]*J_p*nu;

        for (int i=0; i<8; i+=2){
            for (int j=0; j<8; j+=2){
                for (int k=0; k<3; ++k){
                    Kex[i/2+4*j/2] += B[k][i]*tildeD[k]*B[k][j];
                }
            }
        }
        for (int i=1; i<8; i+=2){
            for (int j=1; j<8; j+=2){
                for (int k=0; k<3; ++k){
                    Key[(i-1)/2+4*(j-1)/2] += B[k][i]*tildeD[k]*B[k][j];
                }
            }
        }
	}
}

int ASS_MAP_wIwDI_uJuDJ(int wi,int wd,int w_NPE,int w_dof,int ui,int ud,int u_NPE,int u_dof)
{
    PetscInt              ij;
    PETSC_UNUSED PetscInt r,c,nr,nc;

    nr = w_NPE*w_dof;
    nc = u_NPE*u_dof;

    r = w_dof*wi+wd;
    c = u_dof*ui+ud;

    ij = r*nc+c;

    return ij;
}

void FormGradientOperatorQ12D(double Ge[], double coords[])
{
    double gp_xi[4][2];
    double gp_weight[4];
    double Ni_p[4];
    double GNi_p[2][4],GNx_p[2][4];
    double J_p,fac;
    int    IJ;

    /* define quadrature rule */
    ConstructGaussQuadrature2D(2,gp_xi,gp_weight);

    /* evaluate integral */
    for (int p=0; p < 4; p++) {
        ShapeFunctionQ12D_Evaluate(gp_xi[p],Ni_p);
        ShapeFunctionQ12D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ12D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);
        fac = gp_weight[p]*J_p;

        for (int i=0; i<4; i++) { /* u nodes */
            for (int di=0; di<2; di++) { /* u dofs */
                for (int j=0; j<4; j++) {  /* p nodes, p dofs = 1 (ie no loop) */
                    IJ = ASS_MAP_wIwDI_uJuDJ(i,di,4,2,j,0,4,1);
                    Ge[IJ] = Ge[IJ]-GNx_p[di][i]*Ni_p[j]*fac;
                }
            }
        }
    }
}

void FormAdvectOperatorQ12D(double Ce[], double coords[], double vel[2][4], int m, int n)
{
    double gp_xi[4][2];
    double gp_weight[4];
    double Ni_p[4];
    double GNi_p[2][4],GNx_p[2][4];
    double J_p,fac;
    double vx,vy;

    /* define quadrature rule */
    ConstructGaussQuadrature2D(2,gp_xi,gp_weight);

    /* evaluate integral */
    for (int p=0; p<4; p++) {
        ShapeFunctionQ12D_Evaluate(gp_xi[p],Ni_p);
        ShapeFunctionQ12D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ12D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);
        fac = gp_weight[p]*J_p;
        vx = Ni_p[0]*vel[0][0] + Ni_p[1]*vel[0][1] + Ni_p[2]*vel[0][2] + Ni_p[3]*vel[0][3];
        vy = Ni_p[0]*vel[1][0] + Ni_p[1]*vel[1][1] + Ni_p[2]*vel[1][2] + Ni_p[3]*vel[1][3]; 

        for (int i=0; i<4; i++) { 
            for (int j=0; j<4; j++) { 
                Ce[4*i+j] += vx*GNx_p[0][j]*Ni_p[i]*fac+vy*GNx_p[1][j]*Ni_p[i]*fac;
            }
        }
    }
}


void FormAdvectJacobian(double CJ[], double coords[], double vel[2][4], int m, int n)
{
    double gp_xi[4][2];
    double gp_weight[4];
    double Ni_p[4];
    double GNi_p[2][4],GNx_p[2][4];
    double J_p,fac;
    double v[2];
    double gradU[2][2];
    int    IJ;

    /* define quadrature rule */
    ConstructGaussQuadrature2D(2,gp_xi,gp_weight);

    /* evaluate integral */
    for (int p=0; p<4; p++) {
        ShapeFunctionQ12D_Evaluate(gp_xi[p],Ni_p);
        ShapeFunctionQ12D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ12D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);
        fac = gp_weight[p]*J_p;
        v[0] = Ni_p[0]*vel[0][0] + Ni_p[1]*vel[0][1] + Ni_p[2]*vel[0][2] + Ni_p[3]*vel[0][3];
        v[1] = Ni_p[0]*vel[1][0] + Ni_p[1]*vel[1][1] + Ni_p[2]*vel[1][2] + Ni_p[3]*vel[1][3]; 
        gradU[0][0] = GNx_p[0][0]*vel[0][0] + GNx_p[0][1]*vel[0][1] + GNx_p[0][2]*vel[0][2] + GNx_p[0][3]*vel[0][3];
        gradU[0][1] = GNx_p[0][0]*vel[1][0] + GNx_p[0][1]*vel[1][1] + GNx_p[0][2]*vel[1][2] + GNx_p[0][3]*vel[1][3]; 
        gradU[1][0] = GNx_p[1][0]*vel[0][0] + GNx_p[1][1]*vel[0][1] + GNx_p[1][2]*vel[0][2] + GNx_p[1][3]*vel[0][3];
        gradU[1][1] = GNx_p[1][0]*vel[1][0] + GNx_p[1][1]*vel[1][1] + GNx_p[1][2]*vel[1][2] + GNx_p[1][3]*vel[1][3]; 

        for (int i=0; i<4; i++) { 
            for (int j=0; j<4; j++) { 
                IJ = ASS_MAP_wIwDI_uJuDJ(i,0,4,2,j,0,4,2);
                CJ[IJ] += Ni_p[j]*gradU[0][0]*Ni_p[i]*fac+v[0]*GNx_p[0][j]*Ni_p[i]*fac+v[1]*GNx_p[1][j]*Ni_p[i]*fac;
                IJ = ASS_MAP_wIwDI_uJuDJ(i,0,4,2,j,1,4,2);
                CJ[IJ] += Ni_p[j]*gradU[1][0]*Ni_p[i]*fac;                  
                IJ = ASS_MAP_wIwDI_uJuDJ(i,1,4,2,j,0,4,2);
                CJ[IJ] += Ni_p[j]*gradU[0][1]*Ni_p[i]*fac;  
                IJ = ASS_MAP_wIwDI_uJuDJ(i,1,4,2,j,1,4,2);
                CJ[IJ] += Ni_p[j]*gradU[1][1]*Ni_p[i]*fac+v[0]*GNx_p[0][j]*Ni_p[i]*fac+v[1]*GNx_p[1][j]*Ni_p[i]*fac;  
            }
        }
    }
}

void GetExplicitVel(VortexDOF **states, int i, int j, double vel[2][4])
{
    vel[0][0]=states[j][i].u     ;vel[1][0]=states[j][i].v;
    vel[0][1]=states[j+1][i].u   ;vel[1][1]=states[j+1][i].v;
    vel[0][2]=states[j+1][i+1].u ;vel[1][2]=states[j+1][i+1].v;
    vel[0][3]=states[j][i+1].u   ;vel[1][3]=states[j][i+1].v;
}

void FormDivergenceOperatorQ12D(double De[],double coords[])
{
  PetscScalar Ge[2*4*1*4];
  PetscInt    nr_g,nc_g;

  PetscMemzero(Ge,sizeof(Ge));
  FormGradientOperatorQ12D(Ge,coords);

  nr_g = 2*4;
  nc_g = 1*4;

  for (int i=0; i<nr_g; i++) {
    for (int j=0; j<nc_g; j++) {
      De[nr_g*j+i] = Ge[nc_g*i+j];
    }
  }
}

void FormMassMatrixOperatorQ12D(double Me[],double coords[])
{
    double gp_xi[4][2];
    double gp_weight[4];
    double Ni_p[4];
    double GNi_p[2][4],GNx_p[2][4];
    double J_p,fac;

    /* define quadrature rule */
    ConstructGaussQuadrature2D(2,gp_xi,gp_weight);

    /* evaluate integral */
    for (int p=0; p<4; p++) {
        ShapeFunctionQ12D_Evaluate(gp_xi[p],Ni_p);
        ShapeFunctionQ12D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ12D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);
        fac = gp_weight[p]*J_p;

        for (int i=0; i<4; i++) {
            for (int j=0; j<4; j++) {
                Me[4*i+j] += fac*Ni_p[i]*Ni_p[j];
            }
        }
    }
}


void FormScaledMassMatrixOperatorQ12D(double Pe[],double coords[], double(*penalty)(double, double, double[], int), double samples[], int sampleSize)
{
    double gp_xi[4][2];
    double gp_weight[4];
    double Ni_p[4];
    double GNi_p[2][4],GNx_p[2][4];
    double J_p,fac,lambda,xCoord,yCoord;

    /* define quadrature rule */
    ConstructGaussQuadrature2D(2,gp_xi,gp_weight);

    /* evaluate integral */
    for (int p=0; p<4; p++) {
        ShapeFunctionQ12D_Evaluate(gp_xi[p],Ni_p);
        ShapeFunctionQ12D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ12D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);
        fac = gp_weight[p]*J_p;

        xCoord=0;yCoord=0;
        for (int i=0; i<4; i++) {
            xCoord += Ni_p[i]*coords[i*2];
            yCoord += Ni_p[i]*coords[i*2+1];
        }
        lambda = penalty(xCoord,yCoord,samples,sampleSize);

        for (int i=0; i<4; i++) {
            for (int j=0; j<4; j++) {
                Pe[4*i+j] += fac*Ni_p[i]*Ni_p[j]*lambda;
            }
        }
    }
}

void FormMomentumRhsQ12D(double Fe[],double coords[], double t, void(*Forcing)(double, double, double, double[], double[], int), double samples[], int sampleSize)
{
    double gp_xi[25][2];
    double gp_weight[25];
    double Ni_p[4];
    double GNi_p[2][4],GNx_p[2][4];
    double J_p,fac;
    double xCoord,yCoord,f[2]; 

    /* define quadrature rule */
    ConstructGaussQuadrature2D(5,gp_xi,gp_weight);

    /* evaluate integral */
    for (int p = 0; p < 25; p++) {
        ShapeFunctionQ12D_Evaluate(gp_xi[p],Ni_p);
        ShapeFunctionQ12D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ12D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);
        fac = gp_weight[p]*J_p;

        xCoord=0;yCoord=0;
        for (int i=0; i<4; i++) {
            xCoord += Ni_p[i]*coords[i*2];
            yCoord += Ni_p[i]*coords[i*2+1];
        }
        Forcing(xCoord,yCoord,t,f,samples,sampleSize);

        for (int i = 0; i < 4; i++) {
            Fe[2*i]   += fac*Ni_p[i]*f[0];
            Fe[2*i+1] += fac*Ni_p[i]*f[1]; 
        }
    }    
}

void FormContinuityRhsQ12D(double Fe[],double coords[],double hc[])
{
    double gp_xi[4][2];
    double gp_weight[4];
    double Ni_p[4];
    double GNi_p[2][4],GNx_p[2][4];
    double J_p,fac;

    /* define quadrature rule */
    ConstructGaussQuadrature2D(2,gp_xi,gp_weight);

    /* evaluate integral */
    for (int p=0; p<4; p++) {
        ShapeFunctionQ12D_Evaluate(gp_xi[p],Ni_p);
        ShapeFunctionQ12D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ12D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);
        fac = gp_weight[p]*J_p;

    for (int i=0; i<4; i++) Fe[i] -= fac*Ni_p[i]*hc[p];
  }
}

void DMDAGetElementEqnums2D_up(MatStencil s_u[],MatStencil s_p[],int i,int j)
{
    /* velocity */
    int n = 0;
    /* node 0 */
    s_u[n].i = i; s_u[n].j = j; s_u[n].c = 0; n++; /* Vx0 */
    s_u[n].i = i; s_u[n].j = j; s_u[n].c = 1; n++; /* Vy0 */

    s_u[n].i = i; s_u[n].j = j+1; s_u[n].c = 0; n++;
    s_u[n].i = i; s_u[n].j = j+1; s_u[n].c = 1; n++;

    s_u[n].i = i+1; s_u[n].j = j+1; s_u[n].c = 0; n++;
    s_u[n].i = i+1; s_u[n].j = j+1; s_u[n].c = 1; n++;

    s_u[n].i = i+1; s_u[n].j = j; s_u[n].c = 0; n++;
    s_u[n].i = i+1; s_u[n].j = j; s_u[n].c = 1; n++;

    /* pressure */
    n = 0;

    s_p[n].i = i;   s_p[n].j = j;   s_p[n].c = 2; n++; /* P0 */
    s_p[n].i = i;   s_p[n].j = j+1; s_p[n].c = 2; n++;
    s_p[n].i = i+1; s_p[n].j = j+1; s_p[n].c = 2; n++;
    s_p[n].i = i+1; s_p[n].j = j;   s_p[n].c = 2; n++;
}

void DMDAGetElementEqnums1D_up(MatStencil s_u[],int i,int j,int b)
{
    /* velocity */
    int n = 0;
    /* node 0 */
    s_u[n].i = i;   s_u[n].j = j;   s_u[n].c = b; n++; /* Vx0 */
    s_u[n].i = i;   s_u[n].j = j+1; s_u[n].c = b; n++;
    s_u[n].i = i+1; s_u[n].j = j+1; s_u[n].c = b; n++;
    s_u[n].i = i+1; s_u[n].j = j;   s_u[n].c = b; n++;
}

void FormPreconditionerQ1isoQ2(double Qe[],int i, int j)
{
    if ((i+1)%2==1 && (j+1)%2==1){
        Qe[0*4] = 1;
        Qe[1*4] = 0.5;
        Qe[2*4] = 0.25;
        Qe[3*4] = 0.5;
    } else if ((i+1)%2==0 && (j+1)%2==1){
        Qe[0*4+3] = 0.5;
        Qe[1*4+3] = 0.25;
        Qe[2*4+3] = 0.5;
        Qe[3*4+3] = 1.0;        
    } else if ((i+1)%2==1 && (j+1)%2==0){
        Qe[0*4+1] = 0.5;
        Qe[1*4+1] = 1.0;
        Qe[2*4+1] = 0.5;
        Qe[3*4+1] = 0.25;        
    } else if ((i+1)%2==0 && (j+1)%2==0){
        Qe[0*4+2] = 0.25;
        Qe[1*4+2] = 0.5;
        Qe[2*4+2] = 1.0;
        Qe[3*4+2] = 0.5;        
    } else {
        std::cout << "error form preconditioned q1isoq2" << std::endl;
        std::cout << i << " " << j << " " << i%2 << " " << j%2 << std::endl;
    }
}

void FormIntegralOperator(double Oe[], double coords[], double expCoef)
{
    int ngp=5;
    int ngp_2d=pow(ngp,2);
    double gp_xi[ngp_2d][2];
    double gp_weight[ngp_2d];
    double Ni_p[4];
    double GNi_p[2][4],GNx_p[2][4];
    double J_p,fac;
    double xCoord,yCoord;

    /* define quadrature rule */
    ConstructGaussQuadrature2D(ngp,gp_xi,gp_weight);

    /* evaluate integral */
    for (int p=0; p<ngp_2d; p++) {
        ShapeFunctionQ12D_Evaluate(gp_xi[p],Ni_p);
        ShapeFunctionQ12D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ12D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);
        fac = gp_weight[p]*J_p;

        xCoord=0;yCoord=0;
        for (int i=0; i<4; i++) {
            xCoord += Ni_p[i]*coords[i*2];
            yCoord += Ni_p[i]*coords[i*2+1];
        }

        for (int i = 0; i < 4; i++) {
            Oe[2*i]   += pow(xCoord, expCoef)*pow(yCoord, expCoef)*fac*GNx_p[1][i];
            Oe[2*i+1] -= pow(xCoord, expCoef)*pow(yCoord, expCoef)*fac*GNx_p[0][i];
        }
    }
}

void FormIntegralOperator2(double Oe[], double coords[], double expCoef)
{
    int ngp=5;
    int ngp_2d=pow(ngp,2);
    double gp_xi[ngp_2d][2];
    double gp_weight[ngp_2d];
    double Ni_p[4];
    double GNi_p[2][4],GNx_p[2][4];
    double J_p,fac;
    double xCoord,yCoord;

    /* define quadrature rule */
    ConstructGaussQuadrature2D(ngp,gp_xi,gp_weight);

    /* evaluate integral */
    for (int p=0; p<ngp_2d; p++) {
        ShapeFunctionQ12D_Evaluate(gp_xi[p],Ni_p);
        ShapeFunctionQ12D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ12D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);
        fac = gp_weight[p]*J_p;

        xCoord=0;yCoord=0;
        for (int i=0; i<4; i++) {
            xCoord += Ni_p[i]*coords[i*2];
            yCoord += Ni_p[i]*coords[i*2+1];
        }

        for (int i = 0; i < 4; i++) {
            Oe[2*i]   += pow(yCoord, expCoef)*fac*GNx_p[1][i];
            Oe[2*i+1] += pow(xCoord, expCoef)*fac*GNx_p[0][i];
        }
    }
}


void AssembleM(Mat M, DM meshDM)
{
	DM 						 cda;
	Vec 					 coords;
	DMDACoor2d 			     **_coords;
	MatStencil				 u_eqn[4];
	int						 sex,sey,mx,my;
	double 					 Me[4*4];
	double					 el_coords[4*2];

	DMGetCoordinateDM(meshDM, &cda);
	DMGetCoordinatesLocal(meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(meshDM,&mx,&my,NULL);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
			GetElementCoordinates2D(_coords, ei, ej, el_coords);

            PetscMemzero(Me, sizeof(Me));
            FormMassMatrixOperatorQ12D(Me,el_coords);

            DMDAGetElementEqnums1D_up(u_eqn,ei,ej,0);
            MatSetValuesStencil(M,4,u_eqn,4,u_eqn,Me,ADD_VALUES);

            DMDAGetElementEqnums1D_up(u_eqn,ei,ej,1);
            MatSetValuesStencil(M,4,u_eqn,4,u_eqn,Me,ADD_VALUES);
		}
	}
    
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    DMDAVecRestoreArray(cda,coords,&_coords);
};

void AssembleP(Mat P, DM meshDM, double(*penalty)(double,double,double[],int), double samples[], int sampleSize)
{
	DM 						 cda;
	Vec 					 coords;
	DMDACoor2d 			     **_coords;
	MatStencil				 p_eqn[4];
	int						 sex,sey,mx,my;
	double 					 Pe[4*4];
	double					 el_coords[4*2];

	DMGetCoordinateDM(meshDM, &cda);
	DMGetCoordinatesLocal(meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(meshDM,&mx,&my,NULL);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
			GetElementCoordinates2D(_coords, ei, ej, el_coords);

            PetscMemzero(Pe, sizeof(Pe));
            FormScaledMassMatrixOperatorQ12D(Pe,el_coords,penalty,samples,sampleSize);

            DMDAGetElementEqnums1D_up(p_eqn,ei,ej,2);
            MatSetValuesStencil(P,4,p_eqn,4,p_eqn,Pe,ADD_VALUES);
		}
	}
    MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
    DMDAVecRestoreArray(cda,coords,&_coords);
};

void AssembleA(Mat A, DM meshDM, double nu){
	DM 						 cda;
	Vec 					 coords;
	DMDACoor2d 			     **_coords;
	MatStencil				 u_eqn[4];
	int						 sex,sey,mx,my;
	double 					 Aex[4*4],Aey[4*4];
	double					 el_coords[4*2];
    double                   visc[4]={nu,nu,nu,nu};

	DMGetCoordinateDM(meshDM, &cda);
	DMGetCoordinatesLocal(meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(meshDM,&mx,&my,NULL);

    for (int ej=sey; ej<sey+my; ++ej){
        for (int ei=sex; ei<sex+mx; ++ei){
            GetElementCoordinates2D(_coords, ei, ej, el_coords);

            PetscMemzero(Aex, sizeof(Aex));
            PetscMemzero(Aey, sizeof(Aey));
            FormStressOperatorQ12D(Aex,Aey,el_coords,visc);

            DMDAGetElementEqnums1D_up(u_eqn,ei,ej,0);
            MatSetValuesStencil(A,4,u_eqn,4,u_eqn,Aex,ADD_VALUES);

            DMDAGetElementEqnums1D_up(u_eqn,ei,ej,1);
            MatSetValuesStencil(A,4,u_eqn,4,u_eqn,Aey,ADD_VALUES);
        }
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    DMDAVecRestoreArray(cda,coords,&_coords);
}

void AssemblenuA(Mat A, DM meshDM, double(*Visc)(double, double, double[], int), double samples[], int sampleSize)
{
	DM 						 cda;
	Vec 					 coords;
	DMDACoor2d 			     **_coords;
	MatStencil				 u_eqn[4];
	int						 sex,sey,mx,my;
	double 					 Aex[4*4],Aey[4*4];
	double					 el_coords[4*2];

	DMGetCoordinateDM(meshDM, &cda);
	DMGetCoordinatesLocal(meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(meshDM,&mx,&my,NULL);

    for (int ej=sey; ej<sey+my; ++ej){
        for (int ei=sex; ei<sex+mx; ++ei){
            GetElementCoordinates2D(_coords, ei, ej, el_coords);

            PetscMemzero(Aex, sizeof(Aex));
            PetscMemzero(Aey, sizeof(Aey));
            FormStressOperatorQ12Dnu(Aex,Aey,el_coords,Visc, samples, sampleSize);

            DMDAGetElementEqnums1D_up(u_eqn,ei,ej,0);
            MatSetValuesStencil(A,4,u_eqn,4,u_eqn,Aex,ADD_VALUES);

            DMDAGetElementEqnums1D_up(u_eqn,ei,ej,1);
            MatSetValuesStencil(A,4,u_eqn,4,u_eqn,Aey,ADD_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    DMDAVecRestoreArray(cda,coords,&_coords);
};


void AssembleG(Mat G, DM meshDM){
	DM 						 cda;
	Vec 					 coords;
	DMDACoor2d 			     **_coords;
	MatStencil				 u_eqn[4*2];
	MatStencil  			 p_eqn[4];
	int						 sex,sey,mx,my;
	double 					 Ge[8*4];
	double					 el_coords[4*2];

	DMGetCoordinateDM(meshDM, &cda);
	DMGetCoordinatesLocal(meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(meshDM,&mx,&my,NULL);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
			GetElementCoordinates2D(_coords, ei, ej, el_coords);

			PetscMemzero(Ge, sizeof(Ge));
			FormGradientOperatorQ12D(Ge,el_coords);

            DMDAGetElementEqnums2D_up(u_eqn,p_eqn,ei,ej);

            MatSetValuesStencil(G,4*2,u_eqn,4*1,p_eqn,Ge,ADD_VALUES);
		}
	}
    MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY);
    DMDAVecRestoreArray(cda,coords,&_coords);
}

void AssembleQ(Mat Q, DM meshDM){
	DM 						 cda;
	Vec 					 coords;
	DMDACoor2d 			     **_coords;
	MatStencil				 u_eqn[4*2];
	MatStencil  			 p_eqn[4];
	int						 sex,sey,mx,my;
	double 					 De[4*8];
	double					 el_coords[4*2];

	DMGetCoordinateDM(meshDM, &cda);
	DMGetCoordinatesLocal(meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(meshDM,&mx,&my,NULL);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
			GetElementCoordinates2D(_coords, ei, ej, el_coords);


            PetscMemzero(De, sizeof(De));
            FormDivergenceOperatorQ12D(De,el_coords);

            DMDAGetElementEqnums2D_up(u_eqn,p_eqn,ei,ej);

            MatSetValuesStencil(Q,4*1,p_eqn,4*2,u_eqn,De,ADD_VALUES);
		}
	}
    MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);
    DMDAVecRestoreArray(cda,coords,&_coords);
}

void AssembleC(Mat C,DM meshDM,Vec X,ISLocalToGlobalMapping l2gmapping)
{
	DM 						 cda;
	Vec 					 coords,X_local;
	DMDACoor2d 			     **_coords;
	MatStencil				 u_eqn[4];
    VortexDOF                **states;
	int						 sex,sey,mx,my;
	double 					 Ce[4*4];
	double					 el_coords[4*2];
    double                   vel[2][4]={0.0};

	DMGetCoordinateDM(meshDM, &cda);
	DMGetCoordinatesLocal(meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(meshDM,&mx,&my,NULL);
    DMCreateLocalVector(meshDM,&X_local);
    DMGlobalToLocalBegin(meshDM,X,INSERT_VALUES,X_local);
    DMGlobalToLocalEnd(meshDM,X,INSERT_VALUES,X_local);
    DMDAVecGetArray(meshDM,X_local,&states);

    int starts[2],dims[3];
    MatSetLocalToGlobalMapping(C,l2gmapping,l2gmapping);
    DMDAGetGhostCorners(meshDM,&starts[0],&starts[1],NULL,&dims[0],&dims[1],NULL);
    MatSetStencil(C,2,dims,starts,3);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
			GetElementCoordinates2D(_coords, ei, ej, el_coords);

            PetscMemzero(Ce, sizeof(Ce));
            GetExplicitVel(states, ei, ej, vel);
            FormAdvectOperatorQ12D(Ce, el_coords, vel, ei, ej);

            /* insert element matrix into global matrix */
            DMDAGetElementEqnums1D_up(u_eqn,ei,ej,0);
            MatSetValuesStencil(C,4,u_eqn,4,u_eqn,Ce,ADD_VALUES);

            DMDAGetElementEqnums1D_up(u_eqn,ei,ej,1);
            MatSetValuesStencil(C,4,u_eqn,4,u_eqn,Ce,ADD_VALUES);
		}
	}
    MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);

    DMDAVecRestoreArray(cda,coords,&_coords);
    DMDAVecRestoreArray(meshDM,X_local,&states);
    VecDestroy(&X_local);
}


void AssembleJ(Mat J,DM meshDM,Vec X,ISLocalToGlobalMapping l2gmapping)
{
	DM 						 cda;
	Vec 					 coords,X_local;
	DMDACoor2d 			     **_coords;
	MatStencil				 u_eqn[4*2];
	MatStencil  			 p_eqn[4];
    VortexDOF                **states;
	int						 sex,sey,mx,my;
	double 					 CJ[8*8];
	double					 el_coords[4*2];
    double                   vel[2][4]={0.0};

    int starts[2],dims[3];
    MatSetLocalToGlobalMapping(J,l2gmapping,l2gmapping);
    DMDAGetGhostCorners(meshDM,&starts[0],&starts[1],NULL,&dims[0],&dims[1],NULL);
    MatSetStencil(J,2,dims,starts,3);

	DMGetCoordinateDM(meshDM, &cda);
	DMGetCoordinatesLocal(meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(meshDM,&mx,&my,NULL);
    DMCreateLocalVector(meshDM,&X_local);
    DMGlobalToLocalBegin(meshDM,X,INSERT_VALUES,X_local);
    DMGlobalToLocalEnd(meshDM,X,INSERT_VALUES,X_local);
    DMDAVecGetArray(meshDM,X_local,&states);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
			GetElementCoordinates2D(_coords, ei, ej, el_coords);

            PetscMemzero(CJ, sizeof(CJ));
            GetExplicitVel(states, ei, ej, vel);
            FormAdvectJacobian(CJ, el_coords, vel, ei, ej);

            /* insert element matrix into global matrix */
            DMDAGetElementEqnums2D_up(u_eqn,p_eqn,ei,ej);
            MatSetValuesStencil(J,8,u_eqn,8,u_eqn,CJ,ADD_VALUES);
		}
	}
    MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

    DMDAVecRestoreArray(cda,coords,&_coords);
    DMDAVecRestoreArray(meshDM,X_local,&states);
    VecDestroy(&X_local);
}

void AssembleD(Mat D, DM meshDM)
{
	DM 						 cda;
	Vec 					 coords;
	DMDACoor2d 			     **_coords;
	MatStencil				 u_eqn[4*2];
	MatStencil  			 p_eqn[4];
	int						 sex, sey, mx, my;
    double                   Qe[4*4];
    double                   Ie[8*8];
	double					 el_coords[4*2];

	DMGetCoordinateDM(meshDM, &cda);
	DMGetCoordinatesLocal(meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(meshDM,&mx,&my,NULL);

    PetscMemzero(Ie, sizeof(Ie));
    for (int i=0; i<8; ++i){
        Ie[8*i+i] = 1.0;
    }

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
			GetElementCoordinates2D(_coords, ei, ej, el_coords);

            PetscMemzero(Qe, sizeof(Qe));
            FormPreconditionerQ1isoQ2(Qe, ei, ej);

            DMDAGetElementEqnums2D_up(u_eqn,p_eqn,ei,ej);

            MatSetValuesStencil(D,4,p_eqn,4,p_eqn,Qe,INSERT_VALUES);
            MatSetValuesStencil(D,8,u_eqn,8,u_eqn,Ie,INSERT_VALUES);
		}
	}
    
    MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);

    DMDAVecRestoreArray(cda,coords,&_coords);
}

void DMDASetValuesLocalStencil2D_ADD_VALUES(VortexDOF **fields_F,MatStencil u_eqn[],MatStencil p_eqn[],double Fe_u[],double Fe_p[])
{
    int II,J;

    for (int n=0; n<4; n++) {
        II = u_eqn[2*n].i;
        J = u_eqn[2*n].j;

        fields_F[J][II].u = fields_F[J][II].u+Fe_u[2*n];

        II = u_eqn[2*n+1].i;
        J = u_eqn[2*n+1].j;

        fields_F[J][II].v = fields_F[J][II].v+Fe_u[2*n+1];

        II = p_eqn[n].i;
        J = p_eqn[n].j;

        fields_F[J][II].p = fields_F[J][II].p+Fe_p[n];
    }
}

void AssembleF(Vec F,DM meshDM, double t, void(*Forcing)(double, double, double, double[], double[], int),double samples[], int sampleSize)
{
    DM                     cda;
    Vec                    coords;
    DMDACoor2d             **_coords;
    MatStencil             u_eqn[4*2];
    MatStencil             p_eqn[4*1];

    int                    M,N,P;
    int                    sex,sey,mx,my;
    int                    ei,ej;
    double                 Fe[4*2];
    double                 He[4*2];
    double                 el_coords[4*2];
    Vec                    local_F;
    VortexDOF              **ff;

    DMDAGetInfo(meshDM,0,&M,&N,&P,0,0,0, 0,0,0,0,0,0);
    /* setup for coords */
    DMGetCoordinateDM(meshDM,&cda);
    DMGetCoordinatesLocal(meshDM,&coords);
    DMDAVecGetArray(cda,coords,&_coords);

    /* get acces to the vector */
    DMGetLocalVector(meshDM,&local_F);
    VecZeroEntries(local_F);
    DMDAVecGetArray(meshDM,local_F,&ff);
    DMDAGetElementsCorners(meshDM,&sex,&sey,NULL);
    DMDAGetElementsSizes(meshDM,&mx,&my,NULL);

    for (ej = sey; ej < sey+my; ej++) {
        for (ei = sex; ei < sex+mx; ei++) {
            /* get coords for the element */
            GetElementCoordinates2D(_coords,ei,ej,el_coords);

            PetscMemzero(Fe,sizeof(Fe));
            PetscMemzero(He,sizeof(He));

            FormMomentumRhsQ12D(Fe,el_coords,t,Forcing,samples,sampleSize);
            DMDAGetElementEqnums2D_up(u_eqn,p_eqn,ei,ej);

            DMDASetValuesLocalStencil2D_ADD_VALUES(ff,u_eqn,p_eqn,Fe,He);
            }
        }
    DMDAVecRestoreArray(meshDM,local_F,&ff);
    DMLocalToGlobalBegin(meshDM,local_F,ADD_VALUES,F);
    DMLocalToGlobalEnd(meshDM,local_F,ADD_VALUES,F);
    DMRestoreLocalVector(meshDM,&local_F);

    DMDAVecRestoreArray(cda,coords,&_coords);
}

void AssembleIntegralOperator(Vec intVec,DM meshDM, double expCoef)
{
    DM                     cda;
    Vec                    coords;
    Vec                    local_S;
    DMDACoor2d             **_coords;
    MatStencil             u_eqn[4*2];
    MatStencil             p_eqn[4*1];
    PetscInt               sex,sey,mx,my;
    PetscInt               ei,ej;
    VortexDOF              **sf; 
    double                 Oe[4*2];
    double                 He[4*2];
    double                 el_coords[4*2];

    /* setup for coords */
    DMGetCoordinateDM(meshDM,&cda);
    DMGetCoordinatesLocal(meshDM,&coords);
    DMDAVecGetArray(cda,coords,&_coords);

    /* get acces to the vector */
    DMGetLocalVector(meshDM,&local_S);
    VecZeroEntries(local_S);
    DMDAVecGetArray(meshDM,local_S,&sf);
    DMDAGetElementsCorners(meshDM,&sex,&sey,NULL);
    DMDAGetElementsSizes(meshDM,&mx,&my,NULL);

    for (ej = sey; ej < sey+my; ej++) {
        for (ei = sex; ei < sex+mx; ei++) {
            GetElementCoordinates2D(_coords,ei,ej,el_coords);

            PetscMemzero(Oe,sizeof(Oe));
            PetscMemzero(He,sizeof(He));

            FormIntegralOperator(Oe, el_coords,expCoef);
            DMDAGetElementEqnums2D_up(u_eqn,p_eqn,ei,ej);
            DMDASetValuesLocalStencil2D_ADD_VALUES(sf,u_eqn,p_eqn,Oe,He);
        }
    }

    DMDAVecRestoreArray(meshDM,local_S,&sf);
    DMLocalToGlobalBegin(meshDM,local_S,ADD_VALUES,intVec);
    DMLocalToGlobalEnd(meshDM,local_S,ADD_VALUES,intVec);
    DMRestoreLocalVector(meshDM,&local_S);

    DMDAVecRestoreArray(cda,coords,&_coords);
}


void AssembleIntegralOperator2(Vec intVec,DM meshDM, double expCoef)
{
    DM                     cda;
    Vec                    coords;
    Vec                    local_S;
    DMDACoor2d             **_coords;
    MatStencil             u_eqn[4*2];
    MatStencil             p_eqn[4*1];
    PetscInt               sex,sey,mx,my;
    PetscInt               ei,ej;
    VortexDOF              **sf; 
    double                 Oe[4*2];
    double                 He[4*2];
    double                 el_coords[4*2];

    /* setup for coords */
    DMGetCoordinateDM(meshDM,&cda);
    DMGetCoordinatesLocal(meshDM,&coords);
    DMDAVecGetArray(cda,coords,&_coords);

    /* get acces to the vector */
    DMGetLocalVector(meshDM,&local_S);
    VecZeroEntries(local_S);
    DMDAVecGetArray(meshDM,local_S,&sf);
    DMDAGetElementsCorners(meshDM,&sex,&sey,NULL);
    DMDAGetElementsSizes(meshDM,&mx,&my,NULL);

    for (ej = sey; ej < sey+my; ej++) {
        for (ei = sex; ei < sex+mx; ei++) {
            GetElementCoordinates2D(_coords,ei,ej,el_coords);

            PetscMemzero(Oe,sizeof(Oe));
            PetscMemzero(He,sizeof(He));

            FormIntegralOperator2(Oe, el_coords,expCoef);

            DMDAGetElementEqnums2D_up(u_eqn,p_eqn,ei,ej);

            DMDASetValuesLocalStencil2D_ADD_VALUES(sf,u_eqn,p_eqn,Oe,He);
        }
    }

    DMDAVecRestoreArray(meshDM,local_S,&sf);
    DMLocalToGlobalBegin(meshDM,local_S,ADD_VALUES,intVec);
    DMLocalToGlobalEnd(meshDM,local_S,ADD_VALUES,intVec);
    DMRestoreLocalVector(meshDM,&local_S);

    DMDAVecRestoreArray(cda,coords,&_coords);
}

void ApplyBoundaryCondition(Mat Sys, Vec Sol, Vec Rhs, IS boundaryIS)
{
    MatZeroRowsColumnsIS(Sys,boundaryIS,1.0,Sol,Rhs);
}

void Interpolate(Mat M, Vec load, Vec interpolation)
{
	KSP InterpolationOperator;
	PC  InterpolationPC;
	KSPCreate(PETSC_COMM_SELF, &InterpolationOperator);
	KSPSetType(InterpolationOperator, KSPGMRES);
	KSPSetOperators(InterpolationOperator, M, M);
    KSPSetInitialGuessNonzero(InterpolationOperator, PETSC_TRUE);
	// KSPMonitorSet(InterpolationOperator, MyKSPMonitor, NULL, 0);
	KSPSetTolerances(InterpolationOperator, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPGetPC(InterpolationOperator, &InterpolationPC);
	PCSetType(InterpolationPC, PCJACOBI);
	KSPSetFromOptions(InterpolationOperator);

	KSPSolve(InterpolationOperator, load, interpolation);
	KSPDestroy(&InterpolationOperator);
};
