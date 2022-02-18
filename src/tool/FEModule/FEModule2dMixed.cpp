#include <FEModule2dMixed.h>

Mixed2DMesh::Mixed2DMesh(MPI_Comm comm_, int level_, ELEMENT_TYPE etype_, DMBoundaryType btype_): comm(comm_), level(level_), etype(etype_), btype(btype_)
{
    vortex_num_per_row    = pow(2, level+2);
    vortex_num_per_column = pow(2, level+2);
    total_vortex = vortex_num_per_row*vortex_num_per_column;
    total_element = 2*total_vortex;

    switch (btype)
    {
        case DM_BOUNDARY_PERIODIC:
        {
            DMDACreate2d(comm, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, vortex_num_per_column, vortex_num_per_row, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &meshDM);
            break;
        }
        
        default:
        {
            DMDACreate2d(comm, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, vortex_num_per_column, vortex_num_per_row, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &meshDM);
            break;
        }
    }

	DMSetMatType(meshDM, MATAIJ);
	DMSetUp(meshDM);
    DMDASetUniformCoordinates(meshDM, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    MatCreate(comm, &M);
    MatCreate(comm, &G);
    VecCreate(comm, &f);

    MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,3*total_element+2*total_vortex,3*total_element+2*total_vortex);
    MatSetSizes(G,PETSC_DECIDE,PETSC_DECIDE,3*total_element+2*total_vortex,3*total_element+2*total_vortex);
    VecSetSizes(f,PETSC_DECIDE,3*total_element+2*total_vortex);

    MatSetType(M,MATSEQAIJ);
    MatSetType(G,MATSEQAIJ);
    VecSetType(f,VECSEQ);

	MatSeqAIJSetPreallocation(M, 8, NULL);
	MatSeqAIJSetPreallocation(G, 8, NULL);

    MatSetUp(M);
    MatSetUp(G);
    VecSetUp(f);
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

void ShapeFunctionP12D_Evaluate(double xi_p[], double Ni_p[])
{
	double xi  = xi_p[0];
	double eta = xi_p[1];

	Ni_p[0] = 1.0-xi-eta;
	Ni_p[1] = xi;
	Ni_p[2] = eta;
};

void ShapeFunctionP12D_Evaluate_dxi(double GNi[][3])
{
	GNi[0][0] = -1;	
	GNi[0][1] = 1;
	GNi[0][2] = 0;

	GNi[1][0] = -1;	
	GNi[1][1] = 0;
	GNi[1][2] = 1;
}

void ShapeFunctionP12D_Evaluate_dx(double GNi[][3], double GNx[][3], double coords[], double *det_J)
{
	double  J00,J01,J10,J11;
	double  iJ[2][2];
    J00=0;J01=0;J10=0;J11=0;

    J00=J00+GNi[0][0]*coords[2*0]  +GNi[0][1]*coords[2*1]  +GNi[0][2]*coords[2*2];
    J01=J01+GNi[0][0]*coords[2*0+1]+GNi[0][1]*coords[2*1+1]+GNi[0][2]*coords[2*2+1];
    J10=J10+GNi[1][0]*coords[2*0]  +GNi[1][1]*coords[2*1]  +GNi[1][2]*coords[2*2];
    J11=J11+GNi[1][0]*coords[2*0+1]+GNi[1][1]*coords[2*1+1]+GNi[1][2]*coords[2*2+1];

	*det_J = J00*J11 - J01*J10;

	iJ[0][0] = J11/ *det_J;
	iJ[0][1] =-J01/ *det_J;
	iJ[1][0] =-J10/ *det_J;
	iJ[1][1] = J00/ *det_J;

	for (int n=0; n< 3; ++n){
		GNx[0][n] = GNi[0][n]*iJ[0][0]+GNi[1][n]*iJ[0][1];
		GNx[1][n] = GNi[0][n]*iJ[1][0]+GNi[1][n]*iJ[1][1];
	}
}

void FormGradientOperatorP12D(double Ge1[], double Ge2[], double coords[])
{
    double GNi_p[2][3],GNx_p[2][3];
    double tri_coords[6];
    double J_p,fac;

    tri_coords[0]=coords[0];
    tri_coords[1]=coords[1];
    tri_coords[2]=coords[6];
    tri_coords[3]=coords[7];
    tri_coords[4]=coords[2];
    tri_coords[5]=coords[3];

    ShapeFunctionP12D_Evaluate_dxi(GNi_p);
    ShapeFunctionP12D_Evaluate_dx(GNi_p,GNx_p,tri_coords,&J_p);
    fac = 1.0/2.0*J_p;

    for (int i=0; i<3; i++) {
        Ge1[i] += fac*GNx_p[0][i];
        Ge1[3+i] += fac*GNx_p[1][i];
        Ge1[6+i] += fac*GNx_p[1][i];
        Ge1[9+i] += fac*GNx_p[0][i];
    }

    tri_coords[0]=coords[4];
    tri_coords[1]=coords[5];
    tri_coords[2]=coords[2];
    tri_coords[3]=coords[3];
    tri_coords[4]=coords[6];
    tri_coords[5]=coords[7];

    ShapeFunctionP12D_Evaluate_dx(GNi_p,GNx_p,tri_coords,&J_p);
    fac = 1.0/2.0*J_p;

    for (int i=0; i<3; i++) {
        Ge2[i] += fac*GNx_p[0][i];
        Ge2[3+i] += fac*GNx_p[1][i];
        Ge2[6+i] += fac*GNx_p[1][i];
        Ge2[9+i] += fac*GNx_p[0][i];
    }
}

void FormScaledMassMatrixOperatorP12D(double Me1[],double Me2[],double coords[],double E,double nu)
{
    double GNi_p[2][3],GNx_p[2][3];
    double J_p,fac;
    double tri_coords[6];

    tri_coords[0]=coords[0];
    tri_coords[1]=coords[1];
    tri_coords[2]=coords[6];
    tri_coords[3]=coords[7];
    tri_coords[4]=coords[2];
    tri_coords[5]=coords[3];

    ShapeFunctionP12D_Evaluate_dxi(GNi_p);
    ShapeFunctionP12D_Evaluate_dx(GNi_p,GNx_p,tri_coords,&J_p);
    fac = J_p/2.0;

    Me1[3*0+0] += fac/E;
    Me2[3*0+0] += fac/E;

    Me1[3*0+1] += -nu*fac/E;
    Me2[3*0+1] += -nu*fac/E;

    Me1[3*1+0] += -nu*fac/E;
    Me2[3*1+0] += -nu*fac/E;

    Me1[3*1+1] += fac/E;
    Me2[3*1+1] += fac/E;

    Me1[3*2+2] += 2.0*(1.0+nu)*fac/E;
    Me2[3*2+2] += 2.0*(1.0+nu)*fac/E;
}

void FormScaledMassMatrixOperatorP12D(double Me1[],double Me2[],double coords[],void(*Lame)(double,double,double[],double[],int), double samples[], int sampleSize)
{
    double lameConst[2];
    double nuE1[2],nuE2[2];
    double GNi_p[2][3],GNx_p[2][3];
    double J_p,fac,xCoord,yCoord;

    double tri_coords[6];

    tri_coords[0]=coords[0];
    tri_coords[1]=coords[1];
    tri_coords[2]=coords[6];
    tri_coords[3]=coords[7];
    tri_coords[4]=coords[2];
    tri_coords[5]=coords[3];

    ShapeFunctionP12D_Evaluate_dxi(GNi_p);
    ShapeFunctionP12D_Evaluate_dx(GNi_p,GNx_p,tri_coords,&J_p);
    fac = J_p/2.0;

    xCoord = (coords[0]+coords[6]+coords[2])/3.0;
    yCoord = (coords[1]+coords[7]+coords[3])/3.0;
    Lame(xCoord,yCoord,lameConst,samples,sampleSize);
    nuE1[1] = 4*lameConst[1]*(lameConst[1]+lameConst[0])/(lameConst[0]+2*lameConst[1]);
    nuE1[0] = lameConst[0]/(lameConst[0]+2*lameConst[1]);

    xCoord = (coords[4]+coords[2]+coords[6])/3.0;
    yCoord = (coords[5]+coords[3]+coords[7])/3.0;
    Lame(xCoord,yCoord,lameConst,samples,sampleSize);
    nuE2[1] = 4*lameConst[1]*(lameConst[1]+lameConst[0])/(lameConst[0]+2*lameConst[1]);
    nuE2[0] = lameConst[0]/(lameConst[0]+2*lameConst[1]);

    Me1[3*0+0] += fac/nuE1[1];
    Me2[3*0+0] += fac/nuE2[1];

    Me1[3*0+1] += -nuE1[0]*fac/nuE1[1];
    Me2[3*0+1] += -nuE2[0]*fac/nuE2[1];

    Me1[3*1+0] += -nuE1[0]*fac/nuE1[1];
    Me2[3*1+0] += -nuE2[0]*fac/nuE2[1];

    Me1[3*1+1] += fac/nuE1[1];
    Me2[3*1+1] += fac/nuE2[1];

    Me1[3*2+2] += 2.0*(1.0+nuE1[0])*fac/nuE1[1];
    Me2[3*2+2] += 2.0*(1.0+nuE2[0])*fac/nuE2[1];
}

void FormContinuityRhsP12D(double Fe1[],double Fe2[],double coords[],void(*Forcing)(double, double, double[], double[], int), double samples[], int sampleSize)
{
    double GNi_p[2][3],GNx_p[2][3];
    double J_p,fac;
    double f[3][2]; 
    double tri_coords[6];

    tri_coords[0]=coords[0];
    tri_coords[1]=coords[1];
    tri_coords[2]=coords[6];
    tri_coords[3]=coords[7];
    tri_coords[4]=coords[2];
    tri_coords[5]=coords[3];

    ShapeFunctionP12D_Evaluate_dxi(GNi_p);
    ShapeFunctionP12D_Evaluate_dx(GNi_p,GNx_p,tri_coords,&J_p);
    fac = 1.0/6.0*J_p;

    Forcing(coords[0],coords[1],f[0],samples,sampleSize);
    Forcing(coords[6],coords[7],f[1],samples,sampleSize);
    Forcing(coords[2],coords[3],f[2],samples,sampleSize);

    for (int i = 0; i < 3; i++) {
        Fe1[i]   += fac*f[i][0];
        Fe1[3+i] += fac*f[i][1]; 
    }

    Forcing(coords[4],coords[5],f[0],samples,sampleSize);
    Forcing(coords[2],coords[3],f[1],samples,sampleSize);
    Forcing(coords[6],coords[7],f[2],samples,sampleSize);

    for (int i = 0; i < 3; i++) {
        Fe2[i]   += fac*f[i][0];
        Fe2[3+i] += fac*f[i][1]; 
    }
}

void FormIntegralOperatorP1(double Oe1[], double Oe2[], double coords[], double expCoef)
{
    double GNi_p[2][3],GNx_p[2][3];
    double J_p,fac;
    double tri_coords[6];

    tri_coords[0]=coords[0];
    tri_coords[1]=coords[1];
    tri_coords[2]=coords[6];
    tri_coords[3]=coords[7];
    tri_coords[4]=coords[2];
    tri_coords[5]=coords[3];

    ShapeFunctionP12D_Evaluate_dxi(GNi_p);
    ShapeFunctionP12D_Evaluate_dx(GNi_p,GNx_p,tri_coords,&J_p);
    fac = 1.0/6.0*J_p;

    Oe1[0] += pow(coords[1], expCoef)*fac;
    Oe1[0] += pow(coords[7], expCoef)*fac;
    Oe1[0] += pow(coords[3], expCoef)*fac;

    Oe2[0] += pow(coords[5], expCoef)*fac;
    Oe2[0] += pow(coords[3], expCoef)*fac;
    Oe2[0] += pow(coords[7], expCoef)*fac;

    Oe1[1] += pow(coords[0], expCoef)*fac;
    Oe1[1] += pow(coords[6], expCoef)*fac;
    Oe1[1] += pow(coords[2], expCoef)*fac;

    Oe2[1] += pow(coords[4], expCoef)*fac;
    Oe2[1] += pow(coords[2], expCoef)*fac;
    Oe2[1] += pow(coords[6], expCoef)*fac;    
}


void FormIntegralOperatorP1displacement(double Oe1[], double Oe2[], double coords[], double expCoef)
{
    double GNi_p[2][3],GNx_p[2][3];
    double tri_coords[6];
    double J_p,fac;

    tri_coords[0]=coords[0];
    tri_coords[1]=coords[1];
    tri_coords[2]=coords[6];
    tri_coords[3]=coords[7];
    tri_coords[4]=coords[2];
    tri_coords[5]=coords[3];

    ShapeFunctionP12D_Evaluate_dxi(GNi_p);
    ShapeFunctionP12D_Evaluate_dx(GNi_p,GNx_p,tri_coords,&J_p);
    fac = 1.0/6.0*J_p;

    for (int i=0; i<3; i++) {
        Oe1[i] += pow(coords[0], expCoef)*fac*GNx_p[0][i];
        Oe1[i] += pow(coords[6], expCoef)*fac*GNx_p[0][i];
        Oe1[i] += pow(coords[2], expCoef)*fac*GNx_p[0][i];

        Oe2[i] += pow(coords[1], expCoef)*fac*GNx_p[1][i];
        Oe2[i] += pow(coords[7], expCoef)*fac*GNx_p[1][i];
        Oe2[i] += pow(coords[3], expCoef)*fac*GNx_p[1][i];

        // Oe1[i] += pow(coords[1], expCoef)*pow(coords[0], expCoef)*fac*GNx_p[1][i];
        // Oe1[i] += pow(coords[7], expCoef)*pow(coords[6], expCoef)*fac*GNx_p[1][i];
        // Oe1[i] += pow(coords[3], expCoef)*pow(coords[2], expCoef)*fac*GNx_p[1][i];

        // Oe2[i] += pow(coords[1], expCoef)*pow(coords[0], expCoef)*fac*GNx_p[0][i];
        // Oe2[i] += pow(coords[7], expCoef)*pow(coords[6], expCoef)*fac*GNx_p[0][i];
        // Oe2[i] += pow(coords[3], expCoef)*pow(coords[2], expCoef)*fac*GNx_p[0][i];
    }

    tri_coords[0]=coords[4];
    tri_coords[1]=coords[5];
    tri_coords[2]=coords[2];
    tri_coords[3]=coords[3];
    tri_coords[4]=coords[6];
    tri_coords[5]=coords[7];

    ShapeFunctionP12D_Evaluate_dx(GNi_p,GNx_p,tri_coords,&J_p);
    fac = 1.0/6.0*J_p;

    for (int i=0; i<3; i++) {
        Oe1[3+i] += pow(coords[4], expCoef)*fac*GNx_p[0][i];
        Oe1[3+i] += pow(coords[2], expCoef)*fac*GNx_p[0][i];
        Oe1[3+i] += pow(coords[6], expCoef)*fac*GNx_p[0][i];

        Oe2[3+i] += pow(coords[5], expCoef)*fac*GNx_p[1][i];
        Oe2[3+i] += pow(coords[3], expCoef)*fac*GNx_p[1][i];
        Oe2[3+i] += pow(coords[7], expCoef)*fac*GNx_p[1][i];

        // Oe1[3+i] += pow(coords[5], expCoef)*pow(coords[4], expCoef)*fac*GNx_p[1][i];
        // Oe1[3+i] += pow(coords[3], expCoef)*pow(coords[2], expCoef)*fac*GNx_p[1][i];
        // Oe1[3+i] += pow(coords[7], expCoef)*pow(coords[6], expCoef)*fac*GNx_p[1][i];

        // Oe2[3+i] += pow(coords[5], expCoef)*pow(coords[4], expCoef)*fac*GNx_p[0][i];
        // Oe2[3+i] += pow(coords[3], expCoef)*pow(coords[2], expCoef)*fac*GNx_p[0][i];
        // Oe2[3+i] += pow(coords[7], expCoef)*pow(coords[6], expCoef)*fac*GNx_p[0][i];
    }
}

void AssembleM(Mat M, Mixed2DMesh* mesh, double nu, double Elasticity)
{
	DM 						 cda;
	Vec 					 coords;
	DMDACoor2d 			     **_coords;
	int						 sex,sey,mx,my,eli,elj;
    int                      elementIdx[2];
	double					 el_coords[4*2];
    double                   Me1[3*3],Me2[3*3];

	DMGetCoordinateDM(mesh->meshDM, &cda);
	DMGetCoordinatesLocal(mesh->meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(mesh->meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(mesh->meshDM,&mx,&my,NULL);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
			GetElementCoordinates2D(_coords, ei, ej, el_coords);

            if (ei == -1){
                eli = mesh->vortex_num_per_row-1;
            } else {
                eli = ei;
            }
            if (ej == -1){ 
                elj = mesh->vortex_num_per_column-1;
            } else {
                elj = ej;
            }

            elementIdx[0] = 2*(mesh->vortex_num_per_row*elj + eli);
            elementIdx[1] = elementIdx[0]+1;

            PetscMemzero(Me1, sizeof(Me1));
            PetscMemzero(Me2, sizeof(Me2));
            FormScaledMassMatrixOperatorP12D(Me1,Me2,el_coords,Elasticity,nu);

            MatSetValue(M,elementIdx[0],elementIdx[0],Me1[3*0+0],ADD_VALUES);
            MatSetValue(M,elementIdx[1],elementIdx[1],Me2[3*0+0],ADD_VALUES);

            MatSetValue(M,elementIdx[0],mesh->total_element+elementIdx[0],Me1[3*0+1],ADD_VALUES);
            MatSetValue(M,elementIdx[1],mesh->total_element+elementIdx[1],Me2[3*0+1],ADD_VALUES);

            MatSetValue(M,mesh->total_element+elementIdx[0],elementIdx[0],Me1[3*1+0],ADD_VALUES);
            MatSetValue(M,mesh->total_element+elementIdx[1],elementIdx[1],Me2[3*1+0],ADD_VALUES);

            MatSetValue(M,mesh->total_element+elementIdx[0],mesh->total_element+elementIdx[0],Me1[3*1+1],ADD_VALUES);
            MatSetValue(M,mesh->total_element+elementIdx[1],mesh->total_element+elementIdx[1],Me2[3*1+1],ADD_VALUES);

            MatSetValue(M,2*mesh->total_element+elementIdx[0],2*mesh->total_element+elementIdx[0],Me1[3*2+2],ADD_VALUES);
            MatSetValue(M,2*mesh->total_element+elementIdx[1],2*mesh->total_element+elementIdx[1],Me2[3*2+2],ADD_VALUES);                    
		}
	}
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    DMDAVecRestoreArray(cda,coords,&_coords);
};


void AssembleM(Mat M, Mixed2DMesh* mesh, void(*Lame)(double,double,double[],double[],int), double samples[], int sampleSize)
{
	DM 						 cda;
	Vec 					 coords;
	DMDACoor2d 			     **_coords;
	int						 sex,sey,mx,my,eli,elj;
    int                      elementIdx[2];
	double					 el_coords[4*2];
    double                   Me1[3*3],Me2[3*3];

	DMGetCoordinateDM(mesh->meshDM, &cda);
	DMGetCoordinatesLocal(mesh->meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(mesh->meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(mesh->meshDM,&mx,&my,NULL);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
			GetElementCoordinates2D(_coords, ei, ej, el_coords);

            if (ei == -1){
                eli = mesh->vortex_num_per_row-1;
            } else {
                eli = ei;
            }
            if (ej == -1){ 
                elj = mesh->vortex_num_per_column-1;
            } else {
                elj = ej;
            }

            elementIdx[0] = 2*(mesh->vortex_num_per_row*elj + eli);
            elementIdx[1] = elementIdx[0]+1;

            PetscMemzero(Me1, sizeof(Me1));
            PetscMemzero(Me2, sizeof(Me2));
            FormScaledMassMatrixOperatorP12D(Me1,Me2,el_coords,Lame,samples,sampleSize);

            MatSetValue(M,elementIdx[0],elementIdx[0],Me1[3*0+0],ADD_VALUES);
            MatSetValue(M,elementIdx[1],elementIdx[1],Me2[3*0+0],ADD_VALUES);

            MatSetValue(M,elementIdx[0],mesh->total_element+elementIdx[0],Me1[3*0+1],ADD_VALUES);
            MatSetValue(M,elementIdx[1],mesh->total_element+elementIdx[1],Me2[3*0+1],ADD_VALUES);

            MatSetValue(M,mesh->total_element+elementIdx[0],elementIdx[0],Me1[3*1+0],ADD_VALUES);
            MatSetValue(M,mesh->total_element+elementIdx[1],elementIdx[1],Me2[3*1+0],ADD_VALUES);

            MatSetValue(M,mesh->total_element+elementIdx[0],mesh->total_element+elementIdx[0],Me1[3*1+1],ADD_VALUES);
            MatSetValue(M,mesh->total_element+elementIdx[1],mesh->total_element+elementIdx[1],Me2[3*1+1],ADD_VALUES);

            MatSetValue(M,2*mesh->total_element+elementIdx[0],2*mesh->total_element+elementIdx[0],Me1[3*2+2],ADD_VALUES);
            MatSetValue(M,2*mesh->total_element+elementIdx[1],2*mesh->total_element+elementIdx[1],Me2[3*2+2],ADD_VALUES);                    
		}
	}
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    DMDAVecRestoreArray(cda,coords,&_coords);
};

void AssembleG(Mat G, Mixed2DMesh* mesh){
	DM 						 cda;
	Vec 					 coords;
	DMDACoor2d 			     **_coords;
	int						 sex,sey,mx,my,eli,elii,elj,eljj;
	double 					 Ge1[12],Ge2[12];
    double                   elementIdxTau[2],elementIdxU1[4],elementIdxU2[4];
	double					 el_coords[4*2];

	DMGetCoordinateDM(mesh->meshDM, &cda);
	DMGetCoordinatesLocal(mesh->meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(mesh->meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(mesh->meshDM,&mx,&my,NULL);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
            GetElementCoordinates2D(_coords, ei, ej, el_coords);
            if (ei == -1){
                eli  = mesh->vortex_num_per_row-1;
                elii = 0;
            } else {
                eli  = ei;
                elii = ei+1; 
            }
            if (ej == -1){ 
                elj  = mesh->vortex_num_per_column-1;
                eljj = 0;
            } else {
                elj  = ej;
                eljj = ej+1;
            }

            elementIdxTau[0] = 2*(mesh->vortex_num_per_row*elj + eli);
            elementIdxTau[1] = elementIdxTau[0]+1;

            elementIdxU1[0] = mesh->vortex_num_per_row*elj+eli;
            elementIdxU1[1] = mesh->vortex_num_per_row*elj+elii;
            elementIdxU1[2] = mesh->vortex_num_per_row*eljj+eli;

            elementIdxU2[0] = mesh->vortex_num_per_row*eljj+elii;
            elementIdxU2[1] = mesh->vortex_num_per_row*eljj+eli; 
            elementIdxU2[2] = mesh->vortex_num_per_row*elj+elii;

            PetscMemzero(Ge1, sizeof(Ge1));
            PetscMemzero(Ge2, sizeof(Ge2));
            FormGradientOperatorP12D(Ge1,Ge2,el_coords);

            for (int i=0; i<3; ++i){
                MatSetValue(G,elementIdxTau[0],3*mesh->total_element+elementIdxU1[i],Ge1[3*0+i],ADD_VALUES);
                MatSetValue(G,elementIdxTau[1],3*mesh->total_element+elementIdxU2[i],Ge2[3*0+i],ADD_VALUES);

                MatSetValue(G,mesh->total_element+elementIdxTau[0],3*mesh->total_element+mesh->total_vortex+elementIdxU1[i],Ge1[3*1+i],ADD_VALUES);
                MatSetValue(G,mesh->total_element+elementIdxTau[1],3*mesh->total_element+mesh->total_vortex+elementIdxU2[i],Ge2[3*1+i],ADD_VALUES);

                MatSetValue(G,2*mesh->total_element+elementIdxTau[0],3*mesh->total_element+elementIdxU1[i],Ge1[3*2+i],ADD_VALUES);
                MatSetValue(G,2*mesh->total_element+elementIdxTau[1],3*mesh->total_element+elementIdxU2[i],Ge2[3*2+i],ADD_VALUES);

                MatSetValue(G,2*mesh->total_element+elementIdxTau[0],3*mesh->total_element+mesh->total_vortex+elementIdxU1[i],Ge1[3*3+i],ADD_VALUES);
                MatSetValue(G,2*mesh->total_element+elementIdxTau[1],3*mesh->total_element+mesh->total_vortex+elementIdxU2[i],Ge2[3*3+i],ADD_VALUES);                
            }             
		}
	}
    MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY);
    DMDAVecRestoreArray(cda,coords,&_coords);
}

void AssembleF(Vec F,Mixed2DMesh* mesh,void(*Forcing)(double,double,double[],double[],int),double samples[], int sampleSize)
{
    DM                     cda;
    Vec                    coords;
    DMDACoor2d             **_coords;
    int                    sex,sey,mx,my,eli,elii,elj,eljj;
    double                 elementIdxU1[3],elementIdxU2[3];
    double                 Fe1[6],Fe2[6];
    double                 el_coords[4*2];

	DMGetCoordinateDM(mesh->meshDM, &cda);
	DMGetCoordinatesLocal(mesh->meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(mesh->meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(mesh->meshDM,&mx,&my,NULL);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
            GetElementCoordinates2D(_coords, ei, ej, el_coords);
            if (ei == -1){
                eli  = mesh->vortex_num_per_row-1;
                elii = 0;
            } else {
                eli  = ei;
                elii = ei+1; 
            }
            if (ej == -1){ 
                elj  = mesh->vortex_num_per_column-1;
                eljj = 0;
            } else {
                elj  = ej;
                eljj = ej+1;
            }

            elementIdxU1[0] = mesh->vortex_num_per_row*elj+eli;
            elementIdxU1[1] = mesh->vortex_num_per_row*elj+elii;
            elementIdxU1[2] = mesh->vortex_num_per_row*eljj+eli;

            elementIdxU2[0] = mesh->vortex_num_per_row*eljj+elii;
            elementIdxU2[1] = mesh->vortex_num_per_row*eljj+eli; 
            elementIdxU2[2] = mesh->vortex_num_per_row*elj+elii;

            PetscMemzero(Fe1, sizeof(Fe1));
            PetscMemzero(Fe2, sizeof(Fe2));
            FormContinuityRhsP12D(Fe1,Fe2,el_coords,Forcing,samples,sampleSize);

            for (int i=0; i<3; ++i){
                VecSetValue(F,3*mesh->total_element+elementIdxU1[i],Fe1[i],ADD_VALUES);
                VecSetValue(F,3*mesh->total_element+elementIdxU2[i],Fe2[i],ADD_VALUES);

                VecSetValue(F,3*mesh->total_element+mesh->total_vortex+elementIdxU1[i],Fe1[3+i],ADD_VALUES);
                VecSetValue(F,3*mesh->total_element+mesh->total_vortex+elementIdxU2[i],Fe2[3+i],ADD_VALUES);
            }             
		}
	}
    VecAssemblyBegin(F);
    VecAssemblyEnd(F);
    DMDAVecRestoreArray(cda,coords,&_coords);
}

void AssembleIntegralOperator(Vec intVec,Mixed2DMesh* mesh,double expCoef)
{
    DM                     cda;
    Vec                    coords;
    DMDACoor2d             **_coords;
    PetscInt               sex,sey,mx,my,eli,elj;
    double                 elementIdxTau[2];
    double                 Oe1[2],Oe2[2];
    double                 el_coords[4*2];

	DMGetCoordinateDM(mesh->meshDM, &cda);
	DMGetCoordinatesLocal(mesh->meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(mesh->meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(mesh->meshDM,&mx,&my,NULL);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
            GetElementCoordinates2D(_coords, ei, ej, el_coords);
            if (ei == -1){
                eli  = mesh->vortex_num_per_row-1;
            } else {
                eli  = ei;
            }
            if (ej == -1){ 
                elj  = mesh->vortex_num_per_column-1;
            } else {
                elj  = ej;
            }
            elementIdxTau[0] = 2*(mesh->vortex_num_per_row*elj + eli);
            elementIdxTau[1] = elementIdxTau[0]+1;

            PetscMemzero(Oe1, sizeof(Oe1));
            PetscMemzero(Oe2, sizeof(Oe2));
            FormIntegralOperatorP1(Oe1,Oe2,el_coords,expCoef);

            VecSetValue(intVec,elementIdxTau[0],Oe1[0],ADD_VALUES);
            VecSetValue(intVec,elementIdxTau[1],Oe2[0],ADD_VALUES);

            VecSetValue(intVec,mesh->total_element+elementIdxTau[0],Oe1[1],ADD_VALUES);
            VecSetValue(intVec,mesh->total_element+elementIdxTau[1],Oe2[1],ADD_VALUES);
		}
	}
    VecAssemblyBegin(intVec);
    VecAssemblyEnd(intVec);
    DMDAVecRestoreArray(cda,coords,&_coords);
}


void AssembleIntegralOperatorObs(Vec intVec,Mixed2DMesh* mesh,double expCoef)
{
    DM                     cda;
    Vec                    coords;
    DMDACoor2d             **_coords;
    PetscInt               sex,sey,mx,my,eli,elj;
    double                 elementIdxTau[2];
    double                 Oe1[2],Oe2[2];
    double                 el_coords[4*2];

	DMGetCoordinateDM(mesh->meshDM, &cda);
	DMGetCoordinatesLocal(mesh->meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(mesh->meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(mesh->meshDM,&mx,&my,NULL);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
            GetElementCoordinates2D(_coords, ei, ej, el_coords);
            if (ei == -1){
                eli  = mesh->vortex_num_per_row-1;
            } else {
                eli  = ei;
            }
            if (ej == -1){ 
                elj  = mesh->vortex_num_per_column-1;
            } else {
                elj  = ej;
            }
            elementIdxTau[0] = 2*(mesh->vortex_num_per_row*elj + eli);
            elementIdxTau[1] = elementIdxTau[0]+1;

            PetscMemzero(Oe1, sizeof(Oe1));
            PetscMemzero(Oe2, sizeof(Oe2));
            FormIntegralOperatorP1(Oe1,Oe2,el_coords,expCoef);

            VecSetValue(intVec,elementIdxTau[0],Oe1[0],ADD_VALUES);
            VecSetValue(intVec,elementIdxTau[1],Oe2[0],ADD_VALUES);
		}
	}
    VecAssemblyBegin(intVec);
    VecAssemblyEnd(intVec);
    DMDAVecRestoreArray(cda,coords,&_coords);
}


void AssembleIntegralOperatorQoi(Vec intVec,Mixed2DMesh* mesh,double expCoef)
{
    DM                     cda;
    Vec                    coords;
    DMDACoor2d             **_coords;
    PetscInt               sex,sey,mx,my,eli,elj;
    double                 elementIdxTau[2];
    double                 Oe1[2],Oe2[2];
    double                 el_coords[4*2];

	DMGetCoordinateDM(mesh->meshDM, &cda);
	DMGetCoordinatesLocal(mesh->meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(mesh->meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(mesh->meshDM,&mx,&my,NULL);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
            GetElementCoordinates2D(_coords, ei, ej, el_coords);
            if (ei == -1){
                eli  = mesh->vortex_num_per_row-1;
            } else {
                eli  = ei;
            }
            if (ej == -1){ 
                elj  = mesh->vortex_num_per_column-1;
            } else {
                elj  = ej;
            }
            elementIdxTau[0] = 2*(mesh->vortex_num_per_row*elj + eli);
            elementIdxTau[1] = elementIdxTau[0]+1;

            PetscMemzero(Oe1, sizeof(Oe1));
            PetscMemzero(Oe2, sizeof(Oe2));
            FormIntegralOperatorP1(Oe1,Oe2,el_coords,expCoef);

            VecSetValue(intVec,mesh->total_element+elementIdxTau[0],Oe1[1],ADD_VALUES);
            VecSetValue(intVec,mesh->total_element+elementIdxTau[1],Oe2[1],ADD_VALUES);
		}
	}
    VecAssemblyBegin(intVec);
    VecAssemblyEnd(intVec);
    DMDAVecRestoreArray(cda,coords,&_coords);
}


void AssembleIntegralOperatorQoi2(Vec intVec,Mixed2DMesh* mesh,double expCoef)
{
    DM                     cda;
    Vec                    coords;
    DMDACoor2d             **_coords;
    PetscInt               sex,sey,mx,my,eli,elii,elj,eljj;
    double                 elementIdxU1[3],elementIdxU2[3];
    double                 Oe1[6],Oe2[6];
    double                 el_coords[4*2];

	DMGetCoordinateDM(mesh->meshDM, &cda);
	DMGetCoordinatesLocal(mesh->meshDM, &coords);
	DMDAVecGetArray(cda, coords, &_coords);
	DMDAGetElementsCorners(mesh->meshDM,&sex,&sey,NULL);
	DMDAGetElementsSizes(mesh->meshDM,&mx,&my,NULL);

	for (int ej=sey; ej<sey+my; ++ej){
		for (int ei=sex; ei<sex+mx; ++ei){
            GetElementCoordinates2D(_coords, ei, ej, el_coords);
            if (ei == -1){
                eli  = mesh->vortex_num_per_row-1;
                elii = 0;
            } else {
                eli  = ei;
                elii = ei+1; 
            }
            if (ej == -1){ 
                elj  = mesh->vortex_num_per_column-1;
                eljj = 0;
            } else {
                elj  = ej;
                eljj = ej+1;
            }

            elementIdxU1[0] = mesh->vortex_num_per_row*elj+eli;
            elementIdxU1[1] = mesh->vortex_num_per_row*elj+elii;
            elementIdxU1[2] = mesh->vortex_num_per_row*eljj+eli;

            elementIdxU2[0] = mesh->vortex_num_per_row*eljj+elii;
            elementIdxU2[1] = mesh->vortex_num_per_row*eljj+eli; 
            elementIdxU2[2] = mesh->vortex_num_per_row*elj+elii;

            PetscMemzero(Oe1, sizeof(Oe1));
            PetscMemzero(Oe2, sizeof(Oe2));
            FormIntegralOperatorP1displacement(Oe1,Oe2,el_coords,expCoef);

            VecSetValue(intVec,3*mesh->total_element+elementIdxU1[0],Oe1[0],ADD_VALUES);
            VecSetValue(intVec,3*mesh->total_element+elementIdxU1[1],Oe1[1],ADD_VALUES);
            VecSetValue(intVec,3*mesh->total_element+elementIdxU1[2],Oe1[2],ADD_VALUES);

            VecSetValue(intVec,3*mesh->total_element+mesh->total_vortex+elementIdxU1[0],Oe2[0],ADD_VALUES);
            VecSetValue(intVec,3*mesh->total_element+mesh->total_vortex+elementIdxU1[1],Oe2[1],ADD_VALUES);
            VecSetValue(intVec,3*mesh->total_element+mesh->total_vortex+elementIdxU1[2],Oe2[2],ADD_VALUES);

            VecSetValue(intVec,3*mesh->total_element+elementIdxU2[0],Oe1[3],ADD_VALUES);
            VecSetValue(intVec,3*mesh->total_element+elementIdxU2[1],Oe1[4],ADD_VALUES);
            VecSetValue(intVec,3*mesh->total_element+elementIdxU2[2],Oe1[5],ADD_VALUES);

            VecSetValue(intVec,3*mesh->total_element+mesh->total_vortex+elementIdxU2[0],Oe2[3],ADD_VALUES);
            VecSetValue(intVec,3*mesh->total_element+mesh->total_vortex+elementIdxU2[1],Oe2[4],ADD_VALUES);
            VecSetValue(intVec,3*mesh->total_element+mesh->total_vortex+elementIdxU2[2],Oe2[5],ADD_VALUES);
		}
	}
    VecAssemblyBegin(intVec);
    VecAssemblyEnd(intVec);
    DMDAVecRestoreArray(cda,coords,&_coords);
}