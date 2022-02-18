#include <iostream>
#include <petsc.h>
#include <mpi.h>

typedef struct{
	double gp_coords[50];
	double f[25];
} GaussPointCoefficients;

typedef struct
{
	double u_dof;
	double v_dof; 
	double p_dof;
} StokesDOF;

typedef struct _p_CellProperties *CellProperties;
struct _p_CellProperties{
	int ncells;
	int mx, my;
	int sex, sey;
	GaussPointCoefficients *gpc;
};


typedef struct _p_Mesh* Mesh;
struct _p_Mesh{
	int level;
	int vortex_num_per_row;
	int sex, sey, Imx, Jmy;
	int gauss_points; 
	double m;

	MPI_Comm comm;

	DM         meshDM;
	DM         cda;
	Vec        coords;
	DMDACoor2d **_coords;
	CellProperties cell_properties;

	Mat  system, D;
	Vec  rhs, solution;
};

void gauleg(int n, double x[], double w[]){
	double m = (n+1.0)/2.0;
	double xm = 0.0;
	double xl = 1.0;
	double z, z1, p1, p2, p3, pp;
	for (int i = 1; i < m+1; ++i){
		z = cos(M_PI*(i-0.25)/(n+0.5));
		while (1){
			p1 = 1.0;
			p2 = 0.0;
			for (int j = 1; j < n+1; ++j){
				p3 = p2;
				p2 = p1;
				p1 = ((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
			}
			pp = n*(z*p1-p2)/(z*z-1.0);
			z1 = z;
			z = z1-p1/pp;
			if (abs(z-z1) < 1e-10){
				break;
			}
		} 
		x[i-1] = xm-xl*z;
		x[n-i] = xm+xl*z;
		w[i-1] = 2.0*xl/((1.0-z*z)*pp*pp);
		w[n-i] = w[i-1];
	}
};

void CellPropertiesCreate(DM meshDM, CellProperties *cell){
	CellProperties cells;
	int mx, my, sex, sey;

	PetscNew(&cells);

	DMDAGetElementsCorners(meshDM, &sex, &sey, NULL);
	DMDAGetElementsSizes(meshDM, &mx, &my, NULL);

	cells->mx = mx;
	cells->my = my;
	cells->ncells = mx*my;
	cells->sex = sex;
	cells->sey = sey;

	PetscMalloc1(mx*my, &cells->gpc);

	*cell = cells;
};

int CellPropertiesDestroy(CellProperties *cell){
	CellProperties cells;
	if (!cell) return 0;
	cells = *cell;
	PetscFree(cells->gpc);
	PetscFree(cells);
	*cell = NULL;
	return 0;
};

void CellPropertiesGetCell(CellProperties C, int II, int J, GaussPointCoefficients **G){
	*G = &C->gpc[(II-C->sex) + (J-C->sey)*C->mx];
}

void GetElementCoords(DMDACoor2d **coords, int i, int j, double el_coords[]){
	el_coords[0] = coords[j][i].x;
	el_coords[1] = coords[j][i].y;

	el_coords[2] = coords[j][i+1].x;
	el_coords[3] = coords[j][i+1].y;

	el_coords[4] = coords[j+1][i+1].x;
	el_coords[5] = coords[j+1][i+1].y;

	el_coords[6] = coords[j+1][i].x;
	el_coords[7] = coords[j+1][i].y;
}

void ConstructGaussQuadrature2D(Mesh L2, double gp_xi[][2], double gp_weight[]){
	int n = 5;

	double x[n];
	double w[n];

	gauleg(n, x, w);
	for (int i = 0; i < n; ++i){
		for (int j = 0; j < n; ++j){
			gp_xi[n*j+i][0] = x[i];
			gp_xi[n*j+i][1] = x[j];
			gp_weight[n*i+j] = w[i]*w[j];		
		}
	}
}

void ShapeFunctionQ1Evaluate(double _xi[], double Ni[]){
	double xi  = _xi[0];
	double eta = _xi[1];

	Ni[0] = 0.25*(1.0-xi)*(1.0-eta);
	Ni[1] = 0.25*(1.0+xi)*(1.0-eta);
	Ni[2] = 0.25*(1.0+xi)*(1.0+eta);
	Ni[3] = 0.25*(1.0-xi)*(1.0+eta); 
}

void ShapeFunctionQ1Evaluate_dxi(double _xi[], double GNi[][2]){
	double xi =  _xi[0];
	double eta = _xi[1];

	GNi[0][0] = -0.25+0.25*eta;	
	GNi[1][0] =  0.25-0.25*eta;
	GNi[2][0] =  0.25+0.25*eta;
	GNi[3][0] = -0.25-0.25*eta;

	GNi[0][1] = -0.25+0.25*xi;	
	GNi[1][1] = -0.25-0.25*xi;
	GNi[2][1] =  0.25+0.25*xi;
	GNi[3][1] =  0.25-0.25*xi;
}

void ShapeFunctionQ1Evaluate_dx(double GNi[][2], double GNx[][2], double coords[], double *det_J){
	double  J00 = 0;
	double  J01 = 0;
	double  J10 = 0;
	double  J11 = 0;
	double  iJ[2][2], JJ[2][2];

	double cx = 0;
	double cy = 0;

	for (int n = 0; n < 4; ++n){
		cx = coords[2*n];
		cy = coords[2*n+1];

		J00=J00+GNi[n][0]*cx;
		J01=J01+GNi[n][0]*cy;
		J10=J10+GNi[n][1]*cx;
		J11=J11+GNi[n][1]*cy;
	}

	JJ[0][0] = J00;
	JJ[0][1] = J01;
	JJ[1][0] = J10;
	JJ[1][1] = J11;

	*det_J = J00*J11 - J01*J10;

	iJ[0][0] = J11 / *det_J;
	iJ[0][1] =-J01 / *det_J;
	iJ[1][0] =-J10 / *det_J;
	iJ[1][1] = J00 / *det_J;

	for (int i = 0; i < 4; ++i){
		for (int j = 0; j < 2; ++j){
			GNx[i][j] = GNi[i][0]*iJ[0][j]+GNi[i][1]*iJ[1][j];
		}
	}
}


void FormStressOperatorQ1(Mesh L2, double Ke[], double coords[]){
	double 				gp_xi[25][2];
	double 				gp_weight[25];
	int 				p, i, j;
	double 				GNi_p[4][2], GNx_p[4][2];
	double 				J_p;
	double 				d_dx_i, d_dy_i;

	const int nvdof = 4;

	ConstructGaussQuadrature2D(L2, gp_xi, gp_weight);

	for (p = 0; p < 25; ++p){
		ShapeFunctionQ1Evaluate_dxi(gp_xi[p], GNi_p);
		ShapeFunctionQ1Evaluate_dx(GNi_p, GNx_p, coords, &J_p);

		for (i = 0; i < nvdof; ++i){
			for (j = 0; j < nvdof; ++j){
				Ke[i*nvdof+j] += gp_weight[p]*(GNx_p[i][0]*GNx_p[j][0]+GNx_p[i][1]*GNx_p[j][1])*J_p;
			}
		}
	}
}

void FormGradientOperatorQ1(Mesh L2, double Ge[], double coords[], int dir){
	double 				gp_xi[25][2];
	double 				gp_weight[25];
	int 				p, i, j;
	double 				Ni_p[4];
	double 				GNi_p[4][2], GNx_p[4][2];
	double 				J_p;
	double 				d_dx_i, d_dy_i;

	const int nvdof = 4;

	ConstructGaussQuadrature2D(L2, gp_xi, gp_weight);

	for (p = 0; p < 25; ++p){
		ShapeFunctionQ1Evaluate(gp_xi[p], Ni_p);
		ShapeFunctionQ1Evaluate_dxi(gp_xi[p], GNi_p);
		ShapeFunctionQ1Evaluate_dx(GNi_p, GNx_p, coords, &J_p);

		for (i = 0; i < nvdof; ++i){
			for (j = 0; j < nvdof; ++j){
				Ge[i*nvdof+j] += gp_weight[p]*GNx_p[j][dir]*Ni_p[i]*J_p;
			}
		}
	}
}

void FormQ1isoQ2Precond(double De[], int location){
	if (location == 0){
		De[0+4*0] = 1.0;
		De[0+4*1] = 0.5;
		De[0+4*3] = 0.5;
	} else if (location == 1){
		De[1+4*1] = 1.0;
		De[1+4*0] = 0.5;
		De[1+4*2] = 0.5;
	} else if (location == 2){
		De[2+4*2] = 1.0;
		De[2+4*1] = 0.5;
		De[2+4*3] = 0.5;		
	} else if (location == 3){
		De[3+4*3] = 1.0;
		De[3+4*0] = 0.5;
		De[3+4*2] = 0.5;	
	}
};

void FormMomentumRhsQ1(Mesh L2, double Fe[],double coords[],GaussPointCoefficients* props)
{
  	double gp_xi[25][2];
  	double gp_weight[25];
  	int    p,i;
  	double Ni_p[4];
  	double GNi_p[4][2],GNx_p[4][2];
  	double J_p;
	
  	/* define quadrature rule */
  	ConstructGaussQuadrature2D(L2, gp_xi,gp_weight);

  	/* evaluate integral */
  	for (p = 0; p < 25; p++) {
  	  	ShapeFunctionQ1Evaluate(gp_xi[p],Ni_p);
  	  	ShapeFunctionQ1Evaluate_dxi(gp_xi[p],GNi_p);
  	  	ShapeFunctionQ1Evaluate_dx(GNi_p,GNx_p,coords,&J_p);
  	  	for (i = 0; i < 4; i++) {
  	  	  	Fe[i]  += gp_weight[p]*Ni_p[i]*props->f[p]*J_p;
  	  	}
  	}
}


void DMDAGetElementEqnums(MatStencil s_u[], MatStencil s_v[], MatStencil s_p[], int i, int j){
	int n = 0;
	s_u[n].i = i;   s_u[n].j = j;   s_u[n].c=0; n++;
	s_u[n].i = i+1; s_u[n].j = j;   s_u[n].c=0; n++;
	s_u[n].i = i+1; s_u[n].j = j+1; s_u[n].c=0; n++;
	s_u[n].i = i;   s_u[n].j = j+1; s_u[n].c=0; n++;

	n = 0;
	s_v[n].i = i;   s_v[n].j = j;   s_v[n].c=1; n++;
	s_v[n].i = i+1; s_v[n].j = j;   s_v[n].c=1; n++;
	s_v[n].i = i+1; s_v[n].j = j+1; s_v[n].c=1; n++;
	s_v[n].i = i;   s_v[n].j = j+1; s_v[n].c=1; n++;

	n = 0;
	s_p[n].i = i;   s_p[n].j = j;   s_p[n].c=2; n++;
	s_p[n].i = i+1; s_p[n].j = j;   s_p[n].c=2; n++;
	s_p[n].i = i+1; s_p[n].j = j+1; s_p[n].c=2; n++;
	s_p[n].i = i;   s_p[n].j = j+1; s_p[n].c=2; n++;	
}

void AssembleStokes(Mesh L2){
	DM 						 Acda;
	Vec 					 Acoords;
	DMDACoor2d 				 **_Acoords;
	MatStencil				 s_u[4], s_v[4], s_p[4];
	int						 sex, sey, mx, my;
	int 					 ei, ej;
	double 					 Ae[16], Be1[16], Be2[16], Be1T[16], Be2T[16];
	double					 el_coords[8];
	GaussPointCoefficients   *props;
	int 					 n, M, N;
	double 					 buffer;

	DMCreateMatrix(L2->meshDM, &L2->system);
	DMDAGetInfo(L2->meshDM, NULL, &M, &N, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL );
	DMGetCoordinateDM(L2->meshDM, &Acda);
	DMGetCoordinatesLocal(L2->meshDM, &Acoords);
	DMDAVecGetArray(Acda, Acoords, &_Acoords);
	DMDAGetElementsCorners(L2->meshDM, &sex, &sey, NULL);
	DMDAGetElementsSizes(L2->meshDM, &mx, &my, NULL);
	
	for (int ej = sey; ej < sey+my; ++ej){
		for (int ei = sex; ei < sex+mx; ++ei){
			GetElementCoords(_Acoords, ei, ej, el_coords);
			CellPropertiesGetCell(L2->cell_properties, ei, ej, &props);
			PetscMemzero(Ae, sizeof(Ae));
			PetscMemzero(Be1, sizeof(Be1));
			PetscMemzero(Be2, sizeof(Be2));
			FormStressOperatorQ1(L2, Ae, el_coords);
			FormGradientOperatorQ1(L2, Be1, el_coords, 0);
			FormGradientOperatorQ1(L2, Be2, el_coords, 1);
			DMDAGetElementEqnums(s_u, s_v, s_p, ei, ej);

			MatSetValuesStencil(L2->system, 4, s_u, 4, s_u, Ae, ADD_VALUES);
			MatSetValuesStencil(L2->system, 4, s_v, 4, s_v, Ae, ADD_VALUES);
			MatSetValuesStencil(L2->system, 4, s_p, 4, s_u, Be1, ADD_VALUES);
			MatSetValuesStencil(L2->system, 4, s_p, 4, s_v, Be2, ADD_VALUES);
			for (int i = 0; i < 4; ++i){
				for (int j = 0; j < 4; ++j){
					Be1T[4*j+i] = Be1[4*i+j];
					Be2T[4*j+i] = Be2[4*i+j];					
				}
			}
			MatSetValuesStencil(L2->system, 4, s_u, 4, s_p, Be1T, ADD_VALUES);
			MatSetValuesStencil(L2->system, 4, s_v, 4, s_p, Be2T, ADD_VALUES);
		}
	}
	MatAssemblyBegin(L2->system, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(L2->system, MAT_FINAL_ASSEMBLY);

	DMDAVecRestoreArray(Acda, Acoords, &_Acoords);
}

void AssembleD(Mesh L2){
	DM 						 Dcda;
	Vec 					 Dcoords;
	DMDACoor2d 				 **_Dcoords;
	MatStencil				 s_u[4], s_v[4], s_p[4];
	int						 sex, sey, mx, my;
	int 					 ei, ej;
	const int                *e;
	double 					 De[4*4], Ie[4*4];
	double					 el_coords[4*2];
	GaussPointCoefficients   *props;
	int 					 n, M, N, dir, pos;

	DMDAGetInfo(L2->meshDM, NULL, &M, &N, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL );
	DMGetCoordinateDM(L2->meshDM, &Dcda);
	DMGetCoordinatesLocal(L2->meshDM, &Dcoords);
	DMDAVecGetArray(Dcda, Dcoords, &_Dcoords);
	DMDAGetElementsCorners(L2->meshDM, &sex, &sey, NULL);
	DMDAGetElementsSizes(L2->meshDM, &mx, &my, NULL);
	DMDASetElementType(Dcda, DMDA_ELEMENT_Q1);

	for (int ej = sey; ej < sey+my; ++ej){
		for (int ei = sex; ei < sex+mx; ++ei){
			GetElementCoords(_Dcoords, ei, ej, el_coords);
			CellPropertiesGetCell(L2->cell_properties, ei, ej, &props);

			PetscMemzero(De, sizeof(De));
			PetscMemzero(Ie, sizeof(Ie));

			if ((ei) % 2 == 0 && (ej) % 2 == 0){
				dir = 0;
			} else if (ei % 2 == 1 && ej % 2 == 0){
				dir = 1;
			} else if (ei % 2 == 1 && ej % 2 == 1){
				dir = 2;
			} else if (ei % 2 == 0 && ej % 2 == 1){
				dir = 3;
			}
			FormQ1isoQ2Precond(De, dir);
			for (int i = 0; i < 4; ++i){
				Ie[i*4+i] = 1;
			}
			DMDAGetElementEqnums(s_u, s_v, s_p, ei, ej);
			MatSetValuesStencil(L2->D, 4, s_p, 4, s_p, De, INSERT_VALUES);
			MatSetValuesStencil(L2->D, 4, s_u, 4, s_u, Ie, INSERT_VALUES);
			MatSetValuesStencil(L2->D, 4, s_v, 4, s_v, Ie, INSERT_VALUES);
		}
	}
	MatAssemblyBegin(L2->D, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(L2->D, MAT_FINAL_ASSEMBLY);
}

void VecSetValuesStencil(StokesDOF **fields_F,MatStencil s_u[],MatStencil s_v[],MatStencil s_p[], double Fe_u[])
{
  	int n,II,J;
	for (n = 0; n<4; n++) {
	  	II = s_u[n].i;
	  	J = s_u[n].j;
  		fields_F[J][II].u_dof = fields_F[J][II].u_dof+Fe_u[n];

	  	II = s_v[n].i;
	  	J = s_v[n].j;
  		fields_F[J][II].v_dof = fields_F[J][II].v_dof+Fe_u[n];

	  	II = s_p[n].i;
	  	J = s_p[n].j;
  		fields_F[J][II].p_dof = 0;
  	}
}


void AssembleF(Mesh L2){
	DM                     Fcda;
	Vec                    Fcoords;
	DMDACoor2d             **_Fcoords;
	MatStencil             s_u[4], s_v[4], s_p[4];
	int 	               sex,sey,mx,my;
	int 	               ei,ej;
	double	               Fe[4];
	double	               el_coords[8];
	GaussPointCoefficients *props;
	Vec                    local_F;
	StokesDOF              **ff;
	int                    n,M,N,P;
	DMDAGetInfo(L2->meshDM, NULL, &M, &N, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL );
	/* setup for coords */
	DMGetCoordinateDM(L2->meshDM,&Fcda);
	DMGetCoordinatesLocal(L2->meshDM,&Fcoords);
	DMDAVecGetArray(Fcda,Fcoords,&_Fcoords);
	/* get acces to the vector */
	DMGetLocalVector(L2->meshDM,&local_F);
	VecZeroEntries(local_F);
	DMDAVecGetArray(L2->meshDM,local_F,&ff);
	DMDAGetElementsCorners(L2->meshDM,&sex,&sey, NULL);
	DMDAGetElementsSizes(L2->meshDM,&mx,&my, NULL);

  	for (ej = sey; ej < sey+my; ej++) {
  	  	for (ei = sex; ei < sex+mx; ei++) {
  	  	  	GetElementCoords(_Fcoords,ei,ej,el_coords);
  	  	  	CellPropertiesGetCell(L2->cell_properties,ei,ej,&props);
  	  	  	PetscMemzero(Fe,sizeof(Fe));
  	  	  	FormMomentumRhsQ1(L2, Fe,el_coords,props);
			DMDAGetElementEqnums(s_u, s_v, s_p, ei, ej);
  	  	  	VecSetValuesStencil(ff,s_u, s_v, s_p, Fe);
  	  	}
  	}
	DMDAVecRestoreArray(L2->meshDM,local_F,&ff);
	DMLocalToGlobalBegin(L2->meshDM,local_F,ADD_VALUES,L2->rhs);
	DMLocalToGlobalEnd(L2->meshDM,local_F,ADD_VALUES,L2->rhs);
	DMRestoreLocalVector(L2->meshDM,&local_F);
	DMDAVecRestoreArray(Fcda,Fcoords,&_Fcoords);
}

void meshAssemble(Mesh L2){

	double gp_xi[25][2];
	double gp_weight[25];
	double el_coords[8];
	double xi_p[2], Ni_p[4];
	double gp_x, gp_y;

	for(int ej = L2->sey; ej < L2->sey + L2->Jmy; ++ej){
		for(int ei = L2->sex; ei < L2->sex + L2->Imx; ++ei){
			GaussPointCoefficients *cell;
			CellPropertiesGetCell(L2->cell_properties, ei, ej, &cell);
			GetElementCoords(L2->_coords, ei, ej, el_coords);
			ConstructGaussQuadrature2D(L2, gp_xi, gp_weight);

			for (int p = 0; p < 25; ++p){
				xi_p[0] = gp_xi[p][0];
				xi_p[1] = gp_xi[p][1];		
				ShapeFunctionQ1Evaluate(xi_p, Ni_p);		
			
				gp_x = gp_y = 0.0;
				for (int n = 0; n < 4; ++n){
					gp_x = gp_x + Ni_p[n]*el_coords[2*n];
					gp_y = gp_y + Ni_p[n]*el_coords[2*n+1];
				}
				cell->gp_coords[2*p]   = gp_x;
				cell->gp_coords[2*p+1] = gp_y;

				cell->f[p] = 100*L2->m*sin(2.0*M_PI*gp_x)*sin(2.0*M_PI*gp_y);
			}
		}
	}
	DMDAVecRestoreArray(L2->cda, L2->coords, &L2->_coords);

	DMCreateMatrix(L2->meshDM, &L2->system);
	DMCreateMatrix(L2->meshDM, &L2->D);
	MatSetFromOptions(L2->system);
	MatSetFromOptions(L2->D);
	MatZeroEntries(L2->system);
	MatZeroEntries(L2->D);
	DMCreateGlobalVector(L2->meshDM, &L2->rhs);
	DMCreateGlobalVector(L2->meshDM, &L2->solution);
	VecSetFromOptions(L2->rhs);
	VecSetFromOptions(L2->solution);
	VecZeroEntries(L2->rhs);
	VecZeroEntries(L2->solution);

	AssembleStokes(L2);
	AssembleD(L2);
	AssembleF(L2);
};

void applyBoundary(Mesh L2){
	DM                     Fcda;
	int 	               sex,sey,mx,my;
	int 	               ei,ej;
	int                    n,M,N,nel,nen;
	int 				   cnt = 0;
	const int              *e;
	int                    rows[4];
	DMDAGetInfo(L2->meshDM, NULL, &M, &N, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL );
	/* setup for coords */
	DMGetCoordinateDM(L2->meshDM,&Fcda);
	DMDAGetElementsCorners(L2->meshDM,&sex,&sey, NULL);
	DMDAGetElementsSizes(L2->meshDM,&mx,&my, NULL);
	DMDAGetElements(Fcda, &nel, &nen, &e);
  	for (ej = sey; ej < sey+my; ej++) {
  	  	for (ei = sex; ei < sex+mx; ei++) {
  	  		if (ei == 0) {
  	  			rows[0] = 3*e[4*cnt];
  	  			rows[1] = 3*e[4*cnt+3];
  	  			rows[2] = 3*e[4*cnt]+1;
  	  			rows[3] = 3*e[4*cnt+3]+1;
  	   	  		MatZeroRows(L2->system, 4, rows, 1.0, L2->solution, L2->rhs);

  	  		}
  	  		if (ej == 0){
  	  			rows[0] = 3*e[4*cnt];
  	  			rows[1] = 3*e[4*cnt+1];
  	  			rows[2] = 3*e[4*cnt]+1;
  	  			rows[3] = 3*e[4*cnt+1]+1;  	  			
	  	  		MatZeroRows(L2->system, 4, rows, 1.0, L2->solution, L2->rhs);
  	  		}
  	  		if (ei == M-2){
  	  			rows[0] = 3*e[4*cnt+1];
  	  			rows[1] = 3*e[4*cnt+2];
  	  			rows[2] = 3*e[4*cnt+1]+1;
  	  			rows[3] = 3*e[4*cnt+2]+1;  	  			
	  	  		MatZeroRows(L2->system, 4, rows, 1.0, L2->solution, L2->rhs);
  	  		}
  	  		if (ej == N-2){
  	  			rows[0] = 3*e[4*cnt+2];
  	  			rows[1] = 3*e[4*cnt+3];
  	  			rows[2] = 3*e[4*cnt+2]+1;
  	  			rows[3] = 3*e[4*cnt+3]+1;  	  			
	  	  		MatZeroRows(L2->system, 4, rows, 1.0, L2->solution, L2->rhs);
  	  		}
  	  		cnt++;
  	  	}
  	}
}

void assembleSystem(Mesh L2, int level_){
	L2->level=level_; 
	L2->m = 1.0;
	L2->vortex_num_per_row = 2*pow(2, L2->level+2)+1;
	L2->gauss_points = 25;
	DMDACreate2d(L2->comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, L2->vortex_num_per_row, L2->vortex_num_per_row, PETSC_DECIDE, PETSC_DECIDE, 3, 1, NULL, NULL, &L2->meshDM);
	DMSetMatType(L2->meshDM, MATAIJ);
	DMSetFromOptions(L2->meshDM);
	DMSetUp(L2->meshDM);
	DMDASetUniformCoordinates(L2->meshDM, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	CellPropertiesCreate(L2->meshDM, &L2->cell_properties);
	DMGetCoordinateDM(L2->meshDM, &L2->cda);
	DMGetCoordinatesLocal(L2->meshDM, &L2->coords);
	DMDAVecGetArray(L2->cda, L2->coords, &L2->_coords);
	DMDAGetElementsCorners(L2->meshDM, &L2->sex, &L2->sey, NULL);
	DMDAGetElementsSizes(L2->meshDM, &L2->Imx, &L2->Jmy, NULL);

	meshAssemble(L2);
	Mat workspace;

	MatMatMult(L2->system, L2->D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &workspace);
	MatTransposeMatMult(L2->D, workspace, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &L2->system);
}