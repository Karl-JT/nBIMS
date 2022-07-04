#include "elasticMixed2dRT.h"

class PeriodicBoundary : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        return ((abs(x[0]) < DOLFIN_EPS || abs(x[1]) < DOLFIN_EPS) && 
        (! ((abs(x[0]) < DOLFIN_EPS && abs(x[1]-1.0) < DOLFIN_EPS ) || (abs(x[0]-1.0) < DOLFIN_EPS && abs(x[1]) < DOLFIN_EPS))) && on_boundary);
    }

    void map(const Array<double>& x, Array<double>& y) const
    {
        if (abs(x[0]-1.0) < DOLFIN_EPS && abs(x[1]-1.0) < DOLFIN_EPS)
        {
            y[0] = x[0] - 1.0;
            y[1] = x[1] - 1.0;
        } else if (abs(x[0] - 1.0) < DOLFIN_EPS) {
            y[0] = x[0] - 1.0;
            y[1] = x[1];
        } else {
            y[0] = x[0];
            y[1] = x[1] - 1.0;
        }
    }
};

mixedPoissonSolver::mixedPoissonSolver(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_):comm(comm_), level(level_),num_term(num_term_), noiseVariance(noiseVariance_){
    nx = pow(2,level+2);
    ny = pow(2,level+2);

    geoMesh = std::make_shared<UnitSquareMesh>(comm_,nx,ny,"left");

    obs = 6.008514619642654;//2.432009397737586; //-1.374767714059374;//-1.725570585139927;
    samples = std::make_unique<double[]>(num_term);

    SolverSetup();
};

void mixedPoissonSolver::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};


void mixedPoissonSolver::LinearSystemSetup()
{
    auto boundary =std::make_shared<PeriodicBoundary>();

    W = std::make_shared<MixedPoisson::FunctionSpace>(geoMesh, boundary);
    a = std::make_shared<MixedPoisson::BilinearForm>(W, W);
    L = std::make_shared<MixedPoisson::LinearForm>(W);

    intObs = std::make_shared<Obs::Form_form1>(geoMesh);
    intU1 = std::make_shared<Qoi::Form_form2>(geoMesh);
    intU2 = std::make_shared<Qoi::Form_form3>(geoMesh);
    intQoi = std::make_shared<Qoi::Form_form1>(geoMesh);

    x = std::make_shared<MeshCoordinates>(geoMesh);

    mu = std::make_shared<LameC0>();
    lda = std::make_shared<LameC1>();

    f1 = std::make_shared<ForceC0>();
    f2 = std::make_shared<ForceC1>();

    L->f1 = f1;
    L->f2 = f2;

    w = std::make_shared<Function>(W);

    A = std::make_shared<Matrix>(comm);
    b = std::make_shared<Vector>(comm);
}

void mixedPoissonSolver::SolverSetup(){
    LinearSystemSetup();
}

void mixedPoissonSolver::solve(bool flag)
{
    mu->m = samples[0];
    lda->m = samples[0];

    a->mu = mu;
    a->lda = lda;

    std::vector< std::shared_ptr< const DirichletBC >> bcs;
    assemble_system(*A, *b, *a, *L, bcs);

    // std::cout << "rank: " << rank << " coefficient: " << samples[0] << std::endl;

    PETScKrylovSolver solver(comm); 
    solver.set_operator(A);
    solver.set_from_options();
    //std::clock_t c_start = std::clock();
    //auto wcts = std::chrono::system_clock::now();
    solver.solve(*(w->vector()), *b);
    //std::clock_t c_end = std::clock();
    //double time_elapsed_ms = (c_end-c_start)/ (double)CLOCKS_PER_SEC;
    //std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
    //std::cout << "wall time " << wctduration.count() << " cpu  time: " << time_elapsed_ms << std::endl;
    //solver.str(1);

    intU1->solution = w;
    intU2->solution = w;

    u1_mean = std::make_shared<Constant>(assemble(*intU1));
    u2_mean = std::make_shared<Constant>(assemble(*intU2));

    intObs->x = x;
    intQoi->x = x;

    intObs->solution = w;
    intQoi->solution = w;

    intQoi->s1 = u1_mean;
    intQoi->s2 = u2_mean;
};

void mixedPoissonSolver::priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag)
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

double mixedPoissonSolver::solve4QoI(){
    return QoiOutput();
};

double mixedPoissonSolver::solve4Obs(){
    return ObsOutput();
};

double mixedPoissonSolver::ObsOutput(){
    double obs;
    obs = assemble(*intObs);
    return obs;	
}

double mixedPoissonSolver::QoiOutput(){
    double qoi;
    qoi = 100*assemble(*intQoi);
    return qoi;
}

double mixedPoissonSolver::lnLikelihood(){
	double obsResult = ObsOutput();
	double lnLikelihood = -0.5/noiseVariance*pow(obsResult-obs,2);
	return lnLikelihood;
}
