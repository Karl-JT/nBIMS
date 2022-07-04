#include "nse2dLag.h"
#include "nseforcing.h"

#define REYN 400.0

static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal Re    = REYN;
  const PetscInt  Ncomp = dim;
  PetscInt        c, d;

  for (c = 0; c < Ncomp; ++c) {
    for (d = 0; d < dim; ++d) {
      f0[c] += u[d] * u_x[c*dim+d];
    }
  }
  f0[0] += u_t[0];
  f0[1] += u_t[1];

  f0[0] -= ( Re*((1.0L/2.0L)*PetscSinReal(2*t + 2*x[0]) + PetscSinReal(2*t + x[0] + x[1]) + PetscCosReal(t + x[0] - x[1])) + 2.0*PetscSinReal(t + x[0])*PetscSinReal(t + x[1]))/Re;
  f0[1] -= (-Re*((1.0L/2.0L)*PetscSinReal(2*t + 2*x[1]) + PetscSinReal(2*t + x[0] + x[1]) + PetscCosReal(t + x[0] - x[1])) + 2.0*PetscCosReal(t + x[0])*PetscCosReal(t + x[1]))/Re;
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal Re    = REYN;
  const PetscInt  Ncomp = dim;
  PetscInt        comp, d;

  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      f1[comp*dim+d] = 1.0/Re * u_x[comp*dim+d];
    }
    f1[comp*dim+comp] -= u[Ncomp];
  }
}

static void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += u_x[d*dim+d];
}

static void f1_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = 0.0;
}

/*
  (psi_i, u_j grad_j u_i) ==> (\psi_i, \phi_j grad_j u_i)
*/
static void g0_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt NcI = dim, NcJ = dim;
  PetscInt fc, gc;
  PetscInt d;

  for (d = 0; d < dim; ++d) {
    g0[d*dim+d] = u_tShift;
  }

  for (fc = 0; fc < NcI; ++fc) {
    for (gc = 0; gc < NcJ; ++gc) {
      g0[fc*NcJ+gc] += u_x[fc*NcJ+gc];
    }
  }
}

/*
  (psi_i, u_j grad_j u_i) ==> (\psi_i, \u_j grad_j \phi_i)
*/
static void g1_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt NcI = dim;
  PetscInt NcJ = dim;
  PetscInt fc, gc, dg;
  for (fc = 0; fc < NcI; ++fc) {
    for (gc = 0; gc < NcJ; ++gc) {
      for (dg = 0; dg < dim; ++dg) {
        /* kronecker delta */
        if (fc == gc) {
          g1[(fc*NcJ+gc)*dim+dg] += u[dg];
        }
      }
    }
  }
}

/* < q, \nabla\cdot u >
   NcompI = 1, NcompJ = dim */
static void g1_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d*dim+d] = 1.0; /* \frac{\partial\phi^{u_d}}{\partial x_d} */
}

/* -< \nabla\cdot v, p >
    NcompI = dim, NcompJ = 1 */
static void g2_up(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d*dim+d] = -1.0; /* \frac{\partial\psi^{u_d}}{\partial x_d} */
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal Re    = REYN;
  const PetscInt  Ncomp = dim;
  PetscInt        compI, d;

  for (compI = 0; compI < Ncomp; ++compI) {
    for (d = 0; d < dim; ++d) {
      g3[((compI*Ncomp+compI)*dim+d)*dim+d] = 1.0/Re;
    }
  }
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx)
{
    PetscSimplePointFunc funcs[2];
    void                *ctxs[2];
    DM                   dm;
    PetscDS              ds;
    PetscReal            ferrors[2];
   
    TSGetDM(ts, &dm);
    DMGetDS(dm, &ds);
    PetscDSGetExactSolution(ds, 0, &funcs[0], &ctxs[0]);
    PetscDSGetExactSolution(ds, 1, &funcs[1], &ctxs[1]);
    DMComputeL2FieldDiff(dm, crtime, funcs, ctxs, u, ferrors);
    PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g]\n", (int) step, (double) crtime, (double) ferrors[0], (double) ferrors[1]);
    return 0;
}

PetscErrorCode u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
    u[0] = time + x[0]*x[0] + x[1]*x[1];
    u[1] = time + 2.0*x[0]*x[0] - 2.0*x[0]*x[1];
    return 0;
}

PetscErrorCode p_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  *p = x[0] + x[1] - 1.0;
  return 0;
}

NSE2dLagSolver::NSE2dLagSolver(MPI_Comm comm_, int level_, int num_term_, double noiseVariance_):comm(comm_), level(level_), num_term(num_term_), noiseVariance(noiseVariance_){
    DM        cdm=dm;
    PetscBool simplex;
    PetscInt  dim;
    PetscFE	  fe[2];

    DMCreate(comm, &dm);
    DMSetType(dm, DMPLEX);
    DMSetFromOptions(dm);
    DMViewFromOptions(dm, NULL, "-dm_view");

    DMGetDimension(dm, &dim);
    DMPlexIsSimplex(dm, &simplex);
    PetscFECreateDefault(comm, dim, dim, simplex, "vel_", PETSC_DEFAULT, &fe[0]);
    PetscObjectSetName((PetscObject) fe[0], "velocity");
    PetscFECreateDefault(comm, dim, 1, simplex, "pres_", PETSC_DEFAULT, &fe[1]);
    PetscFECopyQuadrature(fe[0], fe[1]);    
    PetscObjectSetName((PetscObject) fe[1], "pressure");
    DMSetField(dm, 0, NULL, (PetscObject) fe[0]);
    DMSetField(dm, 1, NULL, (PetscObject) fe[1]);
    DMCreateDS(dm);
    SetupProblem();
    while (cdm){
        PetscObject    pressure;
        MatNullSpace   nsp;

        DMGetField(cdm, 1, NULL, &pressure);
        MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nsp);
        PetscObjectCompose(pressure, "nullspace", (PetscObject) nsp);
        MatNullSpaceDestroy(&nsp);
        DMCopyDisc(dm, cdm);
        DMGetCoarseDM(cdm, &cdm);
    }

	PetscFEDestroy(&fe[0]);
	PetscFEDestroy(&fe[1]);

    DMPlexCreateClosureIndex(dm, NULL);
    DMCreateGlobalVector(dm, &u);
    VecDuplicate(u, &r);

    timeSteps = std::pow(2, level_+1);
    deltaT    = tMax/timeSteps;

    samples   = std::make_unique<double[]>(num_term_);

    SolverSetup();
};

void NSE2dLagSolver::updateGeneratorSeed(double seed_){
	generator.seed(seed_);
};

void NSE2dLagSolver::SolverSetup(){
    TSCreate(comm, &ts);
    TSMonitorSet(ts, MonitorError, NULL, NULL);
    TSSetDM(ts, dm);
    DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL);
    DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, NULL);
    DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, NULL);
    TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);
    TSSetFromOptions(ts);
    DMTSCheckFromOptions(ts, u);
}

void NSE2dLagSolver::SetupProblem()
{
    PetscDS           ds;
    DMLabel           label;
    const PetscInt    id=1;
    PetscInt          dim;

    DMGetDimension(dm, &dim);
    DMGetDS(dm, &ds);
    DMGetLabel(dm, "marker", &label);
    
    PetscDSSetResidual(ds, 0, f0_u, f1_u);
    PetscDSSetResidual(ds, 1, f0_p, f1_p);
    PetscDSSetJacobian(ds, 0, 0, g0_uu, g1_uu, NULL, g3_uu);
    PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_up, NULL);
    PetscDSSetJacobian(ds, 1, 0, NULL, g1_pu, NULL, NULL);
    PetscDSSetExactSolution(ds, 0, u_2d, NULL);
    PetscDSSetExactSolution(ds, 1, p_2d, NULL);
    DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) u_2d, NULL, NULL, NULL);
}

void NSE2dLagSolver::solve(bool flag)
{
    {
        PetscSimplePointFunc    funcs[2];
        PetscDS                 ds;

        DMGetDS(dm, &ds);
        PetscDSGetExactSolution(ds, 0, &funcs[0], NULL);
        PetscDSGetExactSolution(ds, 1, &funcs[1], NULL);
        DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, u);   
    }

    std::cout << "start solving" << std::endl;
    TSSolve(ts, u);
    std::cout << "finish solving" << std::endl;
    VecViewFromOptions(u, NULL, "-sol_vec_view");
};

void NSE2dLagSolver::priorSample(double initialSamples[], PRIOR_DISTRIBUTION flag)
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

// double NSE2dLagSolver::solve4QoI(){
//     return QoiOutput();
// };

// double NSE2dLagSolver::solve4Obs(){
//     return ObsOutput();
// };

// double NSE2dLagSolver::ObsOutput(){
//     double obs;
//     VecDot(intVecObs, X, &obs);
//     obs = 100.*obs;
//     return obs;	
// }

// double NSE2dLagSolver::QoiOutput(){
//     double qoi;
//     VecDot(intVecQoi, X_snap, &qoi);
//     qoi = 100.*qoi;
//     return qoi;
// }

// double NSE2dLagSolver::lnLikelihood(){
// 	double obsResult = ObsOutput();
// 	double lnLikelihood = -0.5/noiseVariance*pow(obsResult-obs,2);
// 	return lnLikelihood;
// }
