#pragma once
#include <dolfin.h>
#include <string>
#include <sys/stat.h>
#include "uflForm/RANSPseudoTimeStepping.h"
#include "uflForm/RANSCostW.h"
#include <random>

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
	double width = 20.0*nozzWidth;
	double height = 8.0*nozzWidth;
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

	MPI_Comm mpi_comm;
	int rank;
	int nprocs;

	class InletBoundary : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return (on_boundary && abs(x[0] - 0.0) < DOLFIN_EPS);}
	};

	class SymmetryBoundary : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return (on_boundary && abs(x[1] - 0.0) < DOLFIN_EPS);}
	};

	class FarfieldBoundary : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return (on_boundary && (abs(x[0] - 20.0) < DOLFIN_EPS || abs(x[1] - 8.0) < DOLFIN_EPS));}
	};

	class InnerDomain : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return x[0] < 20.0 + DOLFIN_EPS;	}
	};
	

	jetFlowGeo(int nx_, int ny_);
	~jetFlowGeo(){};
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

	jetFlowSolver(int nx, int ny);

	class uBoundary : public Expression{
	public:
		uBoundary() : Expression(2){}
		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = 0.5+0.5*tanh((-abs(x[1])+0.5)/0.1);
			values[1] = 0.0;
		}
	};

	class kBoundary : public Expression{
	public:
		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = 0.0002+0.0002*tanh((-abs(x[1])+0.5)/0.1);
		}
	};

	class eBoundary : public Expression{
	public:
		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = 7.2e-7+7.2e-7*tanh((-abs(x[1])+0.5)/0.1);
		}
	};

	class initCond : public Expression {
	public:
		initCond() : Expression(5){}
		void eval(Array<double>& values, const Array<double>& x) const{
	        values[0] = 1.0/(1.0+0.12*x[0])*(0.5+0.5*tanh((-abs(x[1])+0.5*(1.0+0.12*x[0]))/(1.0+0.12*x[0])*10));
	        values[1] = 0.0;
	        values[2] = 0.0;
	        if (abs(x[1]) < (1.0+ 0.12*x[0])/2.0 + DOLFIN_EPS){
	            values[3] = 0.0004/(1.0+0.12*x[0]);
	        }
	        else {
	            values[3] = 0;
	        }
	        values[4] = 0.09*pow(values[3], 2.0)/0.02;
		}
	};

	class initM : public Expression{
	public:
		double coef[2] = {0.5, 0.5};

		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = coef[0]*sin(3.1415926*x[0]/10.0)*cos(3.1415926*x[1]/8.0) + coef[1]*cos(3.1415926*x[0]/10.0)*cos(3.1415926*x[1]/8.0);
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
			values[0] = sin(3.1415926*x[0]/10.0)*cos(3.1415926*x[1]/8.0);
		}		
	};

	class mTran2 : public Expression{
	public:
		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = cos(3.1415926*x[0]/10.0)*cos(3.1415926*x[1]/8.0);
		}		
	};

	std::shared_ptr<jetFlowGeo> geo;

	std::shared_ptr<Function> m;
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

	std::shared_ptr<DirichletBC> u_homogenize;    
	std::shared_ptr<DirichletBC> k_homogenize;    
	std::shared_ptr<DirichletBC> e_homogenize;    	
	std::shared_ptr<DirichletBC> axis_homogenize; 

	std::vector<const DirichletBC*> bcs;  			
	std::vector<const DirichletBC*> bcs0; 

	std::shared_ptr<Function> x;        
	std::shared_ptr<Function> xl;       
	std::shared_ptr<Function> adjoints; 
	std::shared_ptr<Constant> u_ff;     
	
	std::shared_ptr<Constant> sigma = std::make_shared<Constant>(20.0);

	std::shared_ptr<Function> u; 
	std::shared_ptr<Function> p; 
	std::shared_ptr<Function> k; 
	std::shared_ptr<Function> e; 

	std::shared_ptr<Function> subFun1;
	std::shared_ptr<Function> subFun2;

	std::vector<std::shared_ptr<Function>> subFunctions;

	Vector Jm_vec;
	Vector dJdx;
	Vector dJdm;
	std::shared_ptr<Function> d;
	std::shared_ptr<Function> xd;
	std::shared_ptr<Function> diff;

	std::shared_ptr<Matrix> Transpose(const GenericMatrix & A);

	double cut_off = 1;

	void updateParameter(double parameter[], int numCoef);
	void soluFwd();
	void updateInitialState();
	void generateRealization();
	void soluAdj();
	double misfit();
	void grad(double gradientVector[]);
	std::shared_ptr<Matrix> hessian(double hessianMatrixArray[]);
};
