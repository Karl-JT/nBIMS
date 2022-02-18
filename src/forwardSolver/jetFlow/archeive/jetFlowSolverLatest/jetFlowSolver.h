#include <dolfin.h>
#include <string>
#include <sys/stat.h>
#include "../RANS.h"

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
	std::shared_ptr<jetFlowGeo> geo;

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
	        values[0] = 1.0/(1.0+0.12*x[0])*(0.5+0.5*tanh((-abs(x[1])+0.5*(1.0+0.12*x[0]))/(1.0/10.0)));
	        values[1] = 0;
	        values[2] = 0;
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
		void eval(Array<double>& values, const Array<double>& x) const{
			values[0] = 0.0;
		}
	};

	jetFlowSolver(int nx, int ny);

	void setFEM();
	void setPDE();
	void soluFwd();
};
