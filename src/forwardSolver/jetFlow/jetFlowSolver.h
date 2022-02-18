#pragma once

#include <dolfin.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/fem/GenericDofMap.h>
#include <string>
#include <sys/stat.h>
#include <random>

#include "uflForm/RANSPseudoTimeStepping.h"
#include "uflForm/RANSCostW.h"
#include <dataIO.h>

using namespace dolfin;

class jetFlowGeo{
private:
public:
	int INLET    = 1;
	int AXIS     = 2;
	int FARFIELD = 3;

	int INNER    = 1;
	int OUTER    = 2;

	double nozzWidth = 1.0;
	double width = 30.0*nozzWidth;
	double height = 10.0*nozzWidth;
	double box[4] = {0.0, 0.0, width, height};

	int nx;
	int ny;

	double point0[2] = {0, 0};
	double point1[2] = {width, height};

	Point P0;
	Point P1;

	std::shared_ptr<RectangleMesh> geoMesh;
	std::shared_ptr<MeshFunction<size_t>> boundaryParts;
	std::shared_ptr<MeshFunction<size_t>> domainParts;

	MPI_Comm comm;
	int rank   = 0;
	int nprocs = 1;

	class InletBoundary : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return (on_boundary && abs(x[0] - 0.0) < DOLFIN_EPS);}
	};

	class SymmetryBoundary : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return (on_boundary && abs(x[1] - 0.0) < DOLFIN_EPS);}
	};

	class upperBoundary : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return (on_boundary && abs(x[1] - 10.0) < DOLFIN_EPS);}
	};

	class FarfieldBoundary : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return (on_boundary && abs(x[0] - 30.0) < DOLFIN_EPS);}
	};

	class InnerDomain : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return x[0] < 30.0 + DOLFIN_EPS;	}
	};
	

	jetFlowGeo(int nx_, int ny_);
	~jetFlowGeo(){};

	void meshScale();
};

class jetFlowSolver{
private:
    double nu            = 1e-4;
    double C_mu          = 0.09;
    double sigma_k       = 1.00;
    double sigma_e       = 1.30;
    double C_e1          = 1.44;
    double C_e2          = 1.92;

public:
	int level;
	std::string data_name = "states";
	std::string data_file = "states";

	jetFlowSolver(int nx_, int ny_, int level_);

	class uBoundary : public Expression{
	public:
		double yInput[201];
		double uInput[201];
		double vInput[201];
		uBoundary() : Expression(2){
			txt2read(yInput, 201, "y.csv");
			txt2read(uInput, 201, "u.csv");		
			txt2read(vInput, 201, "v.csv");
		}
		void eval(Array<double>& values, const Array<double>& x) const{
			for (int i = 0; i < 201; ++i){
				if (x[1]-yInput[i] < 0.001){
					values[0] = uInput[i];
					values[1] = vInput[i];
					break;
				} else if(x[1] < yInput[i]){
					values[0] = uInput[i-1] + (uInput[i] - uInput[i-1])/(yInput[i] - yInput[i-1])*(x[1] - yInput[i-1]);
					values[1] = vInput[i-1] + (vInput[i] - vInput[i-1])/(yInput[i] - yInput[i-1])*(x[1] - yInput[i-1]);
					break;
				}
			}
		}
	};

	class kBoundary : public Expression{
	public:
		double yInput[201];
		double kInput[201];
		kBoundary(){
			txt2read(yInput, 201, "y.csv");
			txt2read(kInput, 201, "k.csv");			
		}
		void eval(Array<double>& values, const Array<double>& x) const{
			for (int i = 0; i < 201; ++i){
				if (x[1]-yInput[i] < 0.01){
					values[0] = kInput[i];
					break;
				} else if(x[1] < yInput[i]){
					values[0] = kInput[i-1] + (kInput[i] - kInput[i-1])/(yInput[i] - yInput[i-1])*(x[1] - yInput[i-1]);
					break;
				}
			}
		}
	};

	class eBoundary : public Expression{
	public:
		double yInput[201];
		double eInput[201];
		eBoundary(){
			txt2read(yInput, 201, "y.csv");
			txt2read(eInput, 201, "e.csv");
		}
		void eval(Array<double>& values, const Array<double>& x) const{
			for (int i = 0; i < 201; ++i){
				if (x[1]-yInput[i] < 0.01){
					values[0] = eInput[i];
					break;
				} else if(x[1] < yInput[i]){
					values[0] = eInput[i-1] + (eInput[i] - eInput[i-1])/(yInput[i] - yInput[i-1])*(x[1] - yInput[i-1]);
					break;
				}
			}
		}
	};

	class initCond : public Expression {
	public:
		initCond() : Expression(5){}
		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = 0;//(1/(1+0.06*x[0]))*(0.5+0.5*tanh((-abs(x[1])+0.5*(1+0.06*x[0]))/0.1/(1+0.06*x[0])));
	        values[1] = 0;
	        values[2] = 0;
	        values[3] = 0;//0.0004/(1+0.06*x[0]);
	        values[4] = 0;//7.2e-7/(1+0.06*x[0]);
		}
	};

	class initM : public Expression{
	public:
		double coef[2] = {0.0, 0.0};

		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = coef[0]*sin(3.1415926*x[0]/15.0)*sin(3.1415926*x[1]/5.0) + coef[1]*sin(3.1415926*x[0]/15.0)*cos(3.1415926*x[1]/5.0);
		}
	};

	// class mTran : public Expression{
	// public:
	// 	mTran() : Expression(2){}
	// 	void eval(Array<double>& values, const Array<double>& x) const{
	// 		values[0] = sin(3.1415926*x[0]/10.0)*cos(3.1415926*x[1]/8.0);
	//  		values[1] = cos(3.1415926*x[0]/10.0)*cos(3.1415926*x[1]/8.0);
	// 	}		
	// };

	class mTran1 : public Expression{
	public:
		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = sin(3.1415926*x[0]/15.0)*sin(3.1415926*x[1]/5.0);
		}		
	};

	class mTran2 : public Expression{
	public:
		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = sin(3.1415926*x[0]/15.0)*cos(3.1415926*x[1]/5.0);
		}		
	};
	
	std::shared_ptr<BoundingBoxTree> bbt;

	std::shared_ptr<jetFlowGeo> geo;

	std::shared_ptr<RANSPseudoTimeStepping::CoefficientSpace_x> Vh_STATE;
	std::shared_ptr<RANSPseudoTimeStepping::CoefficientSpace_m> Q;

	std::shared_ptr<RANSPseudoTimeStepping::Form_F> F;
	std::shared_ptr<RANSPseudoTimeStepping::Form_J> J;
	std::shared_ptr<RANSPseudoTimeStepping::Form_J_true> J_true;

	initM mExpression;

	std::shared_ptr<Constant> zero_scaler;
	std::shared_ptr<Constant> zero_vector;
	std::shared_ptr<uBoundary> u_inflow;   
	std::shared_ptr<kBoundary> k_inflow;   
	std::shared_ptr<eBoundary> e_inflow;   

	std::shared_ptr<DirichletBC> u_inflow_boundary; 
	std::shared_ptr<DirichletBC> k_inflow_boundary;
	std::shared_ptr<DirichletBC> e_inflow_boundary;	
	std::shared_ptr<DirichletBC> axis_boundary;  
	std::shared_ptr<DirichletBC> upper_boundary_v;
	std::shared_ptr<DirichletBC> upper_boundary_k;
	std::shared_ptr<DirichletBC> upper_boundary_e;
	std::shared_ptr<DirichletBC> outlet_boundary_v;
	std::shared_ptr<DirichletBC> outlet_boundary_p;

	std::shared_ptr<DirichletBC> momentum_inflow_boundary;
	std::shared_ptr<DirichletBC> momentum_axis_boundary;

	std::shared_ptr<DirichletBC> rans_k_inflow_boundary;
	std::shared_ptr<DirichletBC> rans_e_inflow_boundary;

	std::shared_ptr<DirichletBC> u_homogenize;    
	std::shared_ptr<DirichletBC> k_homogenize;    
	std::shared_ptr<DirichletBC> e_homogenize;    	
	std::shared_ptr<DirichletBC> axis_homogenize; 
	std::shared_ptr<DirichletBC> upper_homogenize_v;
	std::shared_ptr<DirichletBC> upper_homogenize_k;
	std::shared_ptr<DirichletBC> upper_homogenize_e;
	std::shared_ptr<DirichletBC> outlet_homogenize_v;
	std::shared_ptr<DirichletBC> outlet_homogenize_p;

	std::vector<const DirichletBC*> bcs;  			
	std::vector<const DirichletBC*> bcs0; 
	std::vector<const DirichletBC*> bcs_momentum;  			
	std::vector<const DirichletBC*> bcs0_momentum; 
	std::vector<const DirichletBC*> bcs_rans;  			
	std::vector<const DirichletBC*> bcs0_rans; 

	std::shared_ptr<Function> m;
	std::shared_ptr<Function> x;        
	std::shared_ptr<Function> xl;       
	std::shared_ptr<Function> adjoints; 
	std::shared_ptr<Function> momentum;
	std::shared_ptr<Function> rans;
	std::shared_ptr<Constant> u_ff;  

	std::shared_ptr<Constant> sigma = std::make_shared<Constant>(20.0);

	std::shared_ptr<Function> u; 
	std::shared_ptr<Function> p; 
	std::shared_ptr<Function> k; 
	std::shared_ptr<Function> e; 

	std::shared_ptr<Function> ux; 
	std::shared_ptr<Function> uy; 	

	std::shared_ptr<Function> subFun1;
	std::shared_ptr<Function> subFun2;

	std::vector<std::shared_ptr<Function>> subFunctions;

	Vector Jm_vec;
	Vector dJdx;
	Vector dJdm;

	double obsPoints[15000];
	double obsValues[7500];

	PetscInt global_obsNum;
	PetscInt local_obsNum;

	std::vector<Point> points;
	std::vector<PetscInt> LG;

	std::shared_ptr<Function> d;
	std::shared_ptr<Function> xd;
	std::shared_ptr<Function> diff;

	std::shared_ptr<Matrix> pointwiseTransformMatrix;


	std::shared_ptr<Matrix> dolfinTranspose(const GenericMatrix & A);
	std::shared_ptr<Matrix> dolfinMatMatMult(const GenericMatrix & A, const GenericMatrix & B);
	std::shared_ptr<Matrix> dolfinMatTransposeMatMult(const GenericMatrix & A, const GenericMatrix & B);

	void updateParameter(double parameter[], int numCoef);
	// void uncoupledSolFwd();
	void soluFwd();
	void updateInitialState();
	void generateRealization();
	void generatePointRealization();
	void soluAdj();
	double pointwiseMisfit();
	double misfit();
	int pointwiseTransform(double obsPoints[], int obsNum);
	void grad(double gradientVector[]);
	std::shared_ptr<Matrix> hessian(double hessianMatrixArray[]);
};
