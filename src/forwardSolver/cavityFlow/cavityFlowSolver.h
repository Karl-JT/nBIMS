#pragma once

#include <dolfin.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/fem/GenericDofMap.h>
#include <string>
#include <sys/stat.h>
#include <random>

#include "ufl/cavityFlow.h"
#include <dataIO.h>

using namespace dolfin;

class cavityFlowGeo{
private:
public:
	int INTERIOR     = 0;
	int SLIDING_WALL = 1;
	int NOSLIP_WALL  = 2;

	double width  = 1.0;
	double height = 1.0;

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
	int rank;
	int nprocs;

	class TOP : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return (on_boundary && abs(x[1] - 1.0) < DOLFIN_EPS);}
	};

	class BOTTOM : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return (on_boundary && abs(x[1] - 0.0) < DOLFIN_EPS);}
	};

	class LEFT : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return (on_boundary && abs(x[0] - 0.0) < DOLFIN_EPS);}
	};

	class RIGHT : public SubDomain {
		bool inside(const Array<double>& x, bool on_boundary) const
		{ return (on_boundary && abs(x[0] - 1.0) < DOLFIN_EPS);}
	};
	
	cavityFlowGeo(int nx_, int ny_);
	~cavityFlowGeo(){};
};


class cavityFlowSolver{
private:
	double Reynolds = 1.0;
	double nu = 1./Reynolds;

public:
	class initM : public Expression{
	public:
		double coef[2] = {0.0, 0.0};

		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = coef[0]*sin(3.1415926*x[0]/0.5)*sin(3.1415926*x[1]/0.5) + coef[1]*sin(3.1415926*x[0]/0.5)*cos(3.1415926*x[1]/0.5);
		}
	};

	class mTran1 : public Expression{
	public:
		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = sin(3.1415926*x[0]/0.5)*sin(3.1415926*x[1]/0.5);
		}		
	};

	class mTran2 : public Expression{
	public:
		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = sin(3.1415926*x[0]/0.5)*cos(3.1415926*x[1]/0.5);
		}		
	};

	class initCond : public Expression {
	public:
		initCond() : Expression(3){}
		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = 0;
	        values[1] = 0;
	        values[2] = 0;
		}
	};

	std::shared_ptr<cavityFlowGeo> geo;

	std::shared_ptr<DirichletBC> left_boundary;
	std::shared_ptr<DirichletBC> right_boundary;
	std::shared_ptr<DirichletBC> top_boundary;
	std::shared_ptr<DirichletBC> bottom_boundary;
	std::shared_ptr<DirichletBC> top_boundary_homo;
	std::vector<const DirichletBC*> bcs;  			
	std::vector<const DirichletBC*> bcs0; 

	std::shared_ptr<cavityFlow::CoefficientSpace_x> Vh_STATE;
	std::shared_ptr<cavityFlow::CoefficientSpace_m> Q;	
	initM mExpression;

	std::shared_ptr<cavityFlow::Form_F> F;
	std::shared_ptr<cavityFlow::Form_J> J;

	std::shared_ptr<Function> m;
	std::shared_ptr<Function> x;        
	std::shared_ptr<Function> xl;       
	std::shared_ptr<Function> adjoints;
	std::shared_ptr<Function> u; 
	std::shared_ptr<Function> p; 
	std::shared_ptr<Function> ux; 
	std::shared_ptr<Function> uy; 	
	std::shared_ptr<Function> d;
	std::shared_ptr<Function> xd;
	std::shared_ptr<Function> diff;

	Vector Jm_vec;
	Vector dJdx;
	Vector dJdm;

	std::shared_ptr<Function> subFun1;
	std::shared_ptr<Function> subFun2;
	std::vector<std::shared_ptr<Function>> subFunctions;

	std::shared_ptr<Constant> sigma = std::make_shared<Constant>(150.0);

	std::shared_ptr<Matrix> dolfinTranspose(const GenericMatrix & A);
	std::shared_ptr<Matrix> dolfinMatMatMult(const GenericMatrix & A, const GenericMatrix & B);
	std::shared_ptr<Matrix> dolfinMatTransposeMatMult(const GenericMatrix & A, const GenericMatrix & B);

	cavityFlowSolver(int nx, int ny);
	void updateParameter(double parameter[], int numCoef);
	void soluFwd();
	void generateRealization();
	void soluAdj();
	double misfit();
	void grad(double gradientVector[]);
	std::shared_ptr<Matrix> hessian(double hessianMatrixArray[]);
};