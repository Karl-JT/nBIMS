#pragma once

#include <memory>
#include <random>
#include <ctime>
#include <chrono>
#include <dolfin.h>
#include <numericalRecipes.h>
#include "MixedPoisson.h"
#include "Obs.h"
#include "Qoi.h"

using namespace dolfin;

enum PRIOR_DISTRIBUTION{ UNIFORM, GAUSSIAN };

class LameC0 : public Expression
{
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] = exp(m*(cos(2*M_PI*x[0])*sin(2*M_PI*x[1])));
    }

public:
    double m = 0;
};

class LameC1 : public Expression
{
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] = exp(m*(sin(2*M_PI*x[0])*cos(2*M_PI*x[1])));
    }

public:
    double m = 0;
};

class ForceC0 : public Expression
{
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] = 500*sin(2*M_PI*x[0]);
    }
};

class ForceC1 : public Expression
{
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] = 500*sin(2*M_PI*x[1]);
    }
};

class mixedPoissonSolver{
private:
	std::shared_ptr<UnitSquareMesh> geoMesh;	
    std::shared_ptr<MixedPoisson::FunctionSpace> W;
    std::shared_ptr<MixedPoisson::BilinearForm> a;
    std::shared_ptr<MixedPoisson::LinearForm> L;

    std::shared_ptr<LameC0> mu;
    std::shared_ptr<LameC1> lda;

    std::shared_ptr<ForceC0> f1;
    std::shared_ptr<ForceC1> f2;

    std::shared_ptr<Function> w;
    std::shared_ptr<Obs::Form_form1> intObs;
    std::shared_ptr<Qoi::Form_form1> intQoi;
    std::shared_ptr<Qoi::Form_form2> intU1;
    std::shared_ptr<Qoi::Form_form3> intU2;

    std::shared_ptr<Constant> u1_mean;
    std::shared_ptr<Constant> u2_mean;

    std::shared_ptr<MeshCoordinates> x;

    std::shared_ptr<Parameters> p;

    std::shared_ptr<Matrix> A;
    std::shared_ptr<Vector> b;


public:
    MPI_Comm comm;

    int level;
	int nx;
	int ny;

	// double point0[2] = {0.0,0.0};
	// double point1[2] = {1.0,1.0};

	// Point P0;
	// Point P1;

    int    num_term;
    int    rank;
    double obs,noiseVariance;
	double beta = 1.0;

    std::unique_ptr<double[]> samples;
    
	std::default_random_engine generator;
	std::normal_distribution<double> normalDistribution{0.0,1.0};
    std::uniform_real_distribution<double> uniformDistribution{-1.0,1.0};

    mixedPoissonSolver(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_);
    ~mixedPoissonSolver(){
        std::cout << "rank " << rank << " solver exisitng" << std::endl;
    };

    void updateGeneratorSeed(double seed_);
    void priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag=GAUSSIAN);

    void LinearSystemSetup();
    void SolverSetup();
    void solve(bool flag=0);
    double solve4Obs();
    double solve4QoI();
    double ObsOutput();
    double QoiOutput();
    double lnLikelihood();
};
