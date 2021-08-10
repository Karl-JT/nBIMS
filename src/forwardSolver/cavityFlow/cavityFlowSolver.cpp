#include "cavityFlowSolver.h"

cavityFlowGeo::cavityFlowGeo(int nx_, int ny_): nx(nx_), ny(ny_), P0(2, point0), P1(2, point1){
	geoMesh = std::make_shared<RectangleMesh>(P0, P1, nx, ny);
	comm = geoMesh->mpi_comm();
	rank = MPI::rank(comm);
	nprocs = MPI::size(comm);

	boundaryParts = std::make_shared<MeshFunction<size_t>>(geoMesh, geoMesh->topology().dim()-1);
	boundaryParts->set_all(0);

	LEFT Gamma_left;
	RIGHT Gamma_right;
	TOP Gamma_top;
	BOTTOM Gamma_bottom;

	Gamma_left.mark(*boundaryParts, 1);
	Gamma_right.mark(*boundaryParts, 2);
	Gamma_top.mark(*boundaryParts, 3);
	Gamma_bottom.mark(*boundaryParts, 4);
}


cavityFlowSolver::cavityFlowSolver(int nx, int ny){
	set_log_level(30);

	geo = std::make_shared<cavityFlowGeo>(nx, ny);

	Vh_STATE = std::make_shared<cavityFlow::CoefficientSpace_x>(geo->geoMesh);
	Q = std::make_shared<cavityFlow::CoefficientSpace_m>(geo->geoMesh);

	F = std::make_shared<cavityFlow::Form_F>(Vh_STATE);
	J = std::make_shared<cavityFlow::Form_J>(Vh_STATE, Vh_STATE);

	x  = std::make_shared<Function>(Vh_STATE);
	xl = std::make_shared<Function>(Vh_STATE);
	adjoints = std::make_shared<Function>(Vh_STATE);
	m = std::make_shared<Function>(Q);
	m->interpolate(mExpression);

	auto zero_vector = std::make_shared<Constant>(0.0, 0.0);
	auto sliding_velocity = std::make_shared<Constant>(1.0, 0.0);
	left_boundary     = std::make_shared<DirichletBC>(Vh_STATE->sub(0), zero_vector, geo->boundaryParts, 1);
	right_boundary    = std::make_shared<DirichletBC>(Vh_STATE->sub(0), zero_vector, geo->boundaryParts, 2);
	top_boundary      = std::make_shared<DirichletBC>(Vh_STATE->sub(0), sliding_velocity, geo->boundaryParts, 3);
	top_boundary_homo = std::make_shared<DirichletBC>(Vh_STATE->sub(0), zero_vector, geo->boundaryParts, 3);
	bottom_boundary   = std::make_shared<DirichletBC>(Vh_STATE->sub(0), zero_vector, geo->boundaryParts, 4);

	bcs = {left_boundary.get(), right_boundary.get(), top_boundary.get(), bottom_boundary.get()};
	bcs0 = {left_boundary.get(), right_boundary.get(), top_boundary_homo.get(), bottom_boundary.get()};

	x        = std::make_shared<Function>(Vh_STATE);
	xl       = std::make_shared<Function>(Vh_STATE);
	adjoints = std::make_shared<Function>(Vh_STATE);

	u = std::make_shared<Function>(Vh_STATE->sub(0)->collapse());
	p = std::make_shared<Function>(Vh_STATE->sub(1)->collapse());

	ux = std::make_shared<Function>(Vh_STATE->sub(0)->sub(0)->collapse());
	uy = std::make_shared<Function>(Vh_STATE->sub(0)->sub(1)->collapse());

	d  = std::make_shared<Function>(Vh_STATE->sub(0)->sub(0)->collapse());
	xd = std::make_shared<Function>(Vh_STATE);
	diff = std::make_shared<Function>(Vh_STATE);

	mTran1 subFun1_expression;
	mTran2 subFun2_expression;
	subFun1 = std::make_shared<Function>(Q);
	subFun2 = std::make_shared<Function>(Q);
	subFun1->interpolate(subFun1_expression);
	subFun2->interpolate(subFun2_expression);
	subFunctions = {subFun1, subFun2};
};


void cavityFlowSolver::updateParameter(double parameter[], int numCoef){
	mExpression.coef[0] = parameter[0];
	mExpression.coef[1] = parameter[1];
};

void cavityFlowSolver::soluFwd(){
	std::cout << "start forward solve" << std::endl;

	initCond zeroInit;
	x->interpolate(zeroInit);
	assign(xl, x);	

	m->interpolate(mExpression);

	F->sigma = sigma;
	F->m     = m;
	F->x     = x;  
	F->xl    = xl;
	F->ds    = geo->boundaryParts;

	J->sigma = sigma;
	J->m    = m;
	J->x    = x;
	J->ds   = geo->boundaryParts;

	Parameters params("nonlinear_variational_solver");
	Parameters newton_params("newton_solver");
	newton_params.add("relative_tolerance", 1e-3);
	newton_params.add("convergence_criterion", "residual");
	newton_params.add("error_on_nonconvergence", false);
	newton_params.add("maximum_iterations", 20);
	params.add(newton_params);

	Vector r;
	double r_norm;

	assemble(r, *F);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->apply(r);
	}
	solve(*F == 0, *x, bcs, *J, params);

	assign(xl, x);
	F->sigma = sigma;
	F->x  = x; F->xl = xl;
	J->sigma = sigma;
	J->x  = x;
	assemble(r, *F);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->apply(r);
	}

	for (int i = 0; i < 100000; i++){
		assign(xl, x);
		F->x  = x; F->xl = xl;
		J->x  = x; 

		assemble(r, *F);
		for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
			bcs0[bcCount]->apply(r);
		}
		r_norm = norm(r, "l2");
	 	solve(*F == 0, *x, bcs, *J, params);

	 	if (r_norm < 1e-6){ 
	 		std::cout << "norm" << r_norm << std::endl;
	 		break;
	 	} 
	}
};

void cavityFlowSolver::generateRealization(){
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 0.01);

	assign({u, p}, x);
	assign({ux, uy}, u);
	assign(d, ux);

	Vector dNoise;
	dNoise.init(d->vector()->size());
	for (unsigned i = 0; i < d->vector()->size(); i++){
		dNoise.setitem(i, distribution(generator));
	}
	d->vector()->axpy(1, dNoise);

	File dFile("d.pvd");
	dFile << *d;
};

double cavityFlowSolver::misfit(){
	std::shared_ptr<Matrix> W = std::make_shared<Matrix>();
	std::shared_ptr<Matrix> Wt = std::make_shared<Matrix>(); 

	cavityFlow::Form_WuuFormAdd Wform(Vh_STATE, Vh_STATE);

	assemble(*W, Wform);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->zero(*W);
	}
	Wt = dolfinTranspose(*W);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->zero(*Wt);
	}
	W = dolfinTranspose(*Wt);

	assign(diff, x);
	assign({u, p}, x);
	assign({ux, uy}, u);
	assign(u, {d, uy});
	assign(xd, {u, p});

	diff->vector()->axpy(-1, *(xd->vector()));
	W->mult(*(diff->vector()) , dJdx);
	dJdx *= 1.0e4;	

	double cost;
	cost = -0.5*diff->vector()->inner(dJdx);

	return cost;
}

void cavityFlowSolver::soluAdj(){
	PETScKrylovSolver solver("gmres", "jacobi");
	solver.parameters.remove("relative_tolerance");
	solver.parameters.add("relative_tolerance", 1e-3);

	J->sigma = sigma;
	J->m     = m;
	J->x     = x;
	J->ds    = geo->boundaryParts;

	std::shared_ptr<Matrix> A = std::make_shared<Matrix>();
	std::shared_ptr<Matrix> At = std::make_shared<Matrix>(); 
	assemble(*A, *J);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->apply(*A);
	}
	At = dolfinTranspose(*A); 
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->zero(*At);
	}
	PETScMatrix Ap = as_type<const PETScMatrix>(*A);
	solver.set_operators(Ap, Ap);	

	solver.solve(*(adjoints->vector()), dJdx, true);
};

void cavityFlowSolver::grad(double gradientVector[]){
	auto Jm = std::make_shared<cavityFlow::Form_Jm>(Q);

	Jm->m        = m; 
	Jm->x        = x;  
	Jm->ds       = geo->boundaryParts;
	Jm->adjoints = adjoints;

	assemble(Jm_vec, *Jm);

	gradientVector[0] = -Jm_vec.inner(*(subFunctions[0]->vector()));
	gradientVector[1] = -Jm_vec.inner(*(subFunctions[1]->vector()));

	std::cout << "gradient vector: " << gradientVector[0] << " " << gradientVector[1] << std::endl;
};

std::shared_ptr<Matrix> cavityFlowSolver::hessian(double hessianMatrixArray[]){
	auto Aform      = std::make_shared<cavityFlow::Form_Aform>(Vh_STATE, Vh_STATE);
	auto Cform      = std::make_shared<cavityFlow::Form_Cform>(Q, Vh_STATE);
	auto WuuForm    = std::make_shared<cavityFlow::Form_WuuForm>(Vh_STATE, Vh_STATE);
	auto WuuFormAdd = std::make_shared<cavityFlow::Form_WuuFormAdd>(Vh_STATE, Vh_STATE);

	auto Jx   = std::make_shared<cavityFlow::Form_Jx>(Vh_STATE);
	auto Jadj = std::make_shared<cavityFlow::Form_Jadj>(Vh_STATE);

	Matrix C;
	Cform->m        = m;
	Cform->x        = x;  
	Cform->ds       = geo->boundaryParts;
	assemble(C, *Cform);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->zero(C);
	}
	PETScMatrix Cp = as_type<const PETScMatrix>(C);

	Matrix A;
	Aform->sigma    = sigma;
	Aform->m        = m;
	Aform->x        = x;  
	Aform->ds       = geo->boundaryParts;
	Jadj->sigma     = sigma;
	Jadj->x         = x;
	Jadj->xl        = xl;
	Jadj->m         = m;
	Jadj->ds        = geo->boundaryParts;
	Vector dummy;
	std::vector<std::shared_ptr<const DirichletBC>> bcs0p;
	bcs0p.push_back(left_boundary); 
	bcs0p.push_back(right_boundary); 
	bcs0p.push_back(top_boundary_homo); 
	bcs0p.push_back(bottom_boundary);
	assemble_system(A, dummy, *Aform, *Jadj, bcs0p);

	auto WuuFinal = std::make_shared<Matrix>(); 
	auto WuuAddT  = std::make_shared<Matrix>(); 
	WuuFormAdd->ds = geo->boundaryParts;
	assemble(*WuuFinal, *WuuFormAdd);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->zero(*WuuFinal);
	}
	WuuAddT = dolfinTranspose(*WuuFinal);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->zero(*WuuAddT);
	}	
	WuuFinal = dolfinTranspose(*WuuAddT);
	(*WuuFinal) *= 1.0e4;

	Vector workSpaceA; Vector workSpaceB;
	auto stateWorkSpace = std::make_shared<Function>(Vh_STATE);
	double term2; double singleEntry;		
	PETScKrylovSolver Asolver("gmres", "jacobi");
	Asolver.parameters.remove("relative_tolerance");
	Asolver.parameters.add("relative_tolerance", 1e-3);
	PETScMatrix Ap = as_type<const PETScMatrix>(A);
	Asolver.set_operators(Ap, Ap);
	Mat hessianMatrix;
    MatCreate(PETSC_COMM_SELF,&hessianMatrix);
    MatSetType(hessianMatrix, MATSEQDENSE);
    MatSetSizes(hessianMatrix,PETSC_DECIDE,PETSC_DECIDE,2,2);
    MatSetFromOptions(hessianMatrix);
    MatSetUp(hessianMatrix);
	for (int i = 0; i < 2; ++i){
		for (int j = 0; j < 2; ++j){ 

			C.mult(*(subFunctions[j]->vector()), workSpaceB);
			Asolver.solve(*(stateWorkSpace->vector()), workSpaceB, false);
			std::cout << "Asolver passed" << std::endl;
			WuuFinal->mult(*(stateWorkSpace->vector()), workSpaceB);
			Asolver.solve(*(stateWorkSpace->vector()), workSpaceB, true);
			std::cout << "Asolver transpose passed" << std::endl;
			C.transpmult(*(stateWorkSpace->vector()), workSpaceA);
			term2 = workSpaceA.inner(*(subFunctions[i]->vector()));	

			singleEntry = term2; // + term1;
			MatSetValue(hessianMatrix, i, j, -singleEntry, INSERT_VALUES);
			hessianMatrixArray[2*i+j] = -singleEntry;
			std::cout << i << " " << j << " " << singleEntry;
		}
	}
	MatAssemblyBegin(hessianMatrix, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(hessianMatrix, MAT_FINAL_ASSEMBLY);

	std::shared_ptr<Matrix> out(new Matrix(PETScMatrix(hessianMatrix)));
	out->str(true);

	return out;	
};

std::shared_ptr<Matrix> cavityFlowSolver::dolfinTranspose(const GenericMatrix & A){
	const PETScMatrix* Ap = &as_type<const PETScMatrix>(A);
	Mat At;
	MatTranspose(Ap->mat(), MAT_INITIAL_MATRIX, &At);

	ISLocalToGlobalMapping rmappingA;
	ISLocalToGlobalMapping cmappingA;
	MatGetLocalToGlobalMapping(Ap->mat(),&rmappingA, &cmappingA);
	MatSetLocalToGlobalMapping(At, cmappingA, rmappingA);

	std::shared_ptr<Matrix> out(new Matrix(PETScMatrix(At)));
	MatDestroy(&At);
	return out;
}

std::shared_ptr<Matrix> cavityFlowSolver::dolfinMatMatMult(const GenericMatrix & A, const GenericMatrix & B)
{
    const PETScMatrix* Ap = &as_type<const PETScMatrix>(A);
    const PETScMatrix* Bp = &as_type<const PETScMatrix>(B);
    Mat CC;
    ::MatMatMult(Ap->mat(), Bp->mat(), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CC);

    ISLocalToGlobalMapping rmappingA;
    ISLocalToGlobalMapping cmappingB;
    MatGetLocalToGlobalMapping(Ap->mat(),&rmappingA,NULL);
    MatGetLocalToGlobalMapping(Bp->mat(),NULL, &cmappingB);

    MatSetLocalToGlobalMapping(CC, rmappingA, cmappingB);

    PETScMatrix CCC = PETScMatrix(CC);
    MatDestroy(&CC);
    return std::shared_ptr<Matrix>( new Matrix(CCC) );
}

std::shared_ptr<Matrix> cavityFlowSolver::dolfinMatTransposeMatMult(const GenericMatrix & A, const GenericMatrix & B)
{
    const PETScMatrix* Ap = &as_type<const PETScMatrix>(A);
    const PETScMatrix* Bp = &as_type<const PETScMatrix>(B);
    Mat CC;
    ::MatTransposeMatMult(Ap->mat(), Bp->mat(), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CC);

    ISLocalToGlobalMapping cmappingA;
    ISLocalToGlobalMapping cmappingB;
    MatGetLocalToGlobalMapping(Ap->mat(),NULL, &cmappingA);
    MatGetLocalToGlobalMapping(Bp->mat(),NULL, &cmappingB);

    MatSetLocalToGlobalMapping(CC, cmappingA, cmappingB);

    PETScMatrix CCC = PETScMatrix(CC);
    MatDestroy(&CC);
    return std::shared_ptr<Matrix>(new Matrix(CCC) );
}