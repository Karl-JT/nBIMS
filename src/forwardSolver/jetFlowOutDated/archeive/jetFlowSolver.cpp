#include "jetFlowSolver.h"

jetFlowGeo::jetFlowGeo(int nx_, int ny_): nx(nx_), ny(ny_), P0(2, point0), P1(2, point1){
	geoMesh = std::make_shared<RectangleMesh>(P0, P1, nx, ny);

	// Scaled mesh
	// auto coord = geoMesh->coordinates();	
	// for (int i = 0; i < coord.size(); ++i){
	// 	if (i % 2 != 0){
	// 		coord[i] = 8.0-8.0*tanh((1.0-coord[i]/8.0)*1.5)/tanh(1.5);
	// 	}
	// }

	boundaryParts = std::make_shared<MeshFunction<size_t>>(geoMesh, geoMesh->topology().dim()-1);
	boundaryParts->set_all(0);

	InletBoundary Gamma_inlet;
	SymmetryBoundary Gamma_axis;;
	FarfieldBoundary Gamma_farfield;

	Gamma_inlet.mark(*boundaryParts, 1);
	Gamma_axis.mark(*boundaryParts, 2);
	Gamma_farfield.mark(*boundaryParts, 3);

	domainParts = std::make_shared<MeshFunction<size_t>>(geoMesh, geoMesh->topology().dim());
	domainParts->set_all(0);

	InnerDomain Beta_inner;
	Beta_inner.mark(*domainParts, INNER);
}

bool fileExists(const std::string& file) {
    struct stat buf;
    return (stat(file.c_str(), &buf) == 0);
}

jetFlowSolver::jetFlowSolver(int nx, int ny){
	geo = std::make_shared<jetFlowGeo>(nx, ny);

	Vh_STATE = std::make_shared<RANSPseudoTimeStepping::CoefficientSpace_x>(geo->geoMesh);
	Q = std::make_shared<RANSPseudoTimeStepping::CoefficientSpace_m>(geo->geoMesh);

	m = std::make_shared<Function>(Vh_STATE->sub(2)->collapse());
	m->interpolate(mExpression);

	F = std::make_shared<RANSPseudoTimeStepping::Form_F>(Vh_STATE);
	J = std::make_shared<RANSPseudoTimeStepping::Form_J>(Vh_STATE, Vh_STATE);

	zero_scaler = std::make_shared<Constant>(0.0);
	zero_vector = std::make_shared<Constant>(0.0, 0.0);
	u_inflow   = std::make_shared<uBoundary>();
	k_inflow   = std::make_shared<kBoundary>();
	e_inflow   = std::make_shared<eBoundary>();

	u_inflow_boundary = std::make_shared<DirichletBC>(Vh_STATE->sub(0), u_inflow, geo->boundaryParts, 1);
	k_inflow_boundary = std::make_shared<DirichletBC>(Vh_STATE->sub(2), k_inflow, geo->boundaryParts, 1);
	e_inflow_boundary = std::make_shared<DirichletBC>(Vh_STATE->sub(3), e_inflow, geo->boundaryParts, 1);	
	axis_boundary     = std::make_shared<DirichletBC>(Vh_STATE->sub(0)->sub(1), zero_scaler, geo->boundaryParts, 2);

	u_homogenize    = std::make_shared<DirichletBC>(Vh_STATE->sub(0), zero_vector, geo->boundaryParts, 1);
	k_homogenize    = std::make_shared<DirichletBC>(Vh_STATE->sub(2), zero_scaler, geo->boundaryParts, 1);
	e_homogenize    = std::make_shared<DirichletBC>(Vh_STATE->sub(3), zero_scaler, geo->boundaryParts, 1);	
	axis_homogenize = std::make_shared<DirichletBC>(Vh_STATE->sub(0)->sub(1), zero_scaler, geo->boundaryParts, 2);

	bcs  = {u_inflow_boundary.get(), k_inflow_boundary.get(), e_inflow_boundary.get(), axis_boundary.get()};			
	bcs0 = {u_homogenize.get(), k_homogenize.get(), e_homogenize.get(), axis_homogenize.get()};

	x        = std::make_shared<Function>(Vh_STATE);
	xl       = std::make_shared<Function>(Vh_STATE);
	adjoints = std::make_shared<Function>(Vh_STATE);
	u_ff     = std::make_shared<Constant>(0.0, 0.0);
	
	sigma = std::make_shared<Constant>(10.0);

	u = std::make_shared<Function>(Vh_STATE->sub(0)->collapse());
	p = std::make_shared<Function>(Vh_STATE->sub(1)->collapse());
	k = std::make_shared<Function>(Vh_STATE->sub(2)->collapse());
	e = std::make_shared<Function>(Vh_STATE->sub(3)->collapse());

	d  = std::make_shared<Function>(Vh_STATE->sub(0)->collapse());
	xd = std::make_shared<Function>(Vh_STATE);
	diff = std::make_shared<Function>(Vh_STATE);	

	subFun1 = std::make_shared<Function>(Q);
	subFun2 = std::make_shared<Function>(Q);
	subFun1->interpolate(subFun1_expression);
	subFun2->interpolate(subFun2_expression);

	subFunctions = {subFun1, subFun2};

};

void jetFlowSolver::soluFwd(){
	std::string data_name = "states";
	std::string data_file = "states.h5";
	if (fileExists(data_file)){
		HDF5File inputFile(geo->geoMesh->mpi_comm(), "states.h5", "r");
		inputFile.read(*x.get(), data_name);
		File uFile("u.pvd");
		File kFile("k.pvd");
		assign({u, p, k, e}, x);
		uFile << *u;
		kFile << *k;
		assign(xl, x);
		inputFile.close();
	} else{
		initCond zeroInit;
		x->interpolate(zeroInit);
		assign(xl, x);
	}

	m->interpolate(mExpression);

	F->sigma = sigma;
	F->u_ff  = u_ff;
	F->m     = m;
	F->x     = x;  
	F->xl    = xl;
	F->ds    = geo->boundaryParts;

	J->sigma = sigma;
	J->m    = m;
	J->x    = x;
	J->xl = xl;
	J->ds   = geo->boundaryParts;

	Parameters params("nonlinear_variational_solver");
	Parameters newton_params("newton_solver");
	newton_params.add("relative_tolerance", 1e-10);
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
	r_norm = norm(r, "l2");
	cout << "norm: " << r_norm << endl;

	solve(*F == 0, *x, bcs, *J, params);

	for (int i = 0; i < 1000; i++){
		assign(xl, x);

		F->x  = x; F->xl = xl;
		J->x  = x; J->xl = xl;

		assemble(r, *F);
		for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
			bcs0[bcCount]->apply(r);
		}
		r_norm = norm(r, "l2");
		cout << "norm: " << r_norm << endl;;

	 	solve(*F == 0, *x, bcs, *J, params);
	 	if (r_norm < 1e-3){
	 		break;
	 	}
	}
	File uFile("u.pvd");
	File kFile("k.pvd");
	assign({u, p, k, e}, x);
	uFile << *u;
	kFile << *k;

	HDF5File outputFile(geo->geoMesh->mpi_comm(), "states.h5", "w");
	outputFile.write(*x.get(), "states");
	outputFile.close();
};

void jetFlowSolver::updateParameter(double parameter[], int numCoef){
	mExpression.coef[0] = parameter[0];
	mExpression.coef[1] = parameter[1];
};

void jetFlowSolver::generateRealization(){
	double noise_level = 0.01;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, noise_level);

	assign(d, u);

	Vector dNoise;
	dNoise.init(d->vector()->size());

	for (unsigned i = 0; i < d->vector()->size(); i++){
		dNoise.setitem(i, distribution(generator));
	}

	d->vector()->axpy(1, dNoise);	

	File dFile("d.pvd");
	dFile << *d;	

	assign(xd, {d, p, k, e});
};


void jetFlowSolver::soluAdj(){
	PETScLUSolver solver;

	J->sigma = sigma;
	J->m    = m;
	J->x    = x;
	J->xl = xl;
	J->ds   = geo->boundaryParts;

	Matrix A;
	assemble(A, *J);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->apply(A);
	}
	PETScMatrix Ap = as_type<const PETScMatrix>(A);
	solver.set_operator(Ap);
	
	std::shared_ptr<Matrix> W = std::make_shared<Matrix>();
	std::shared_ptr<Matrix> Wt = std::make_shared<Matrix>(); 
	auto Wh_STATE = std::make_shared<RANSCostW::Form_Wform_FunctionSpace_1>(geo->geoMesh);
	RANSCostW::Form_Wform Wform(Wh_STATE, Wh_STATE);	

	assemble(*W, Wform);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->zero(*W);
	}
	Wt = Transpose(*W);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->zero(*Wt);
	}
	W = Transpose(*Wt);

	assign(diff, x);
	diff->vector()->axpy(-1, *(xd->vector()));
	W->mult(*(diff->vector()) , dJdx);

	solver.solve(*(adjoints->vector()), dJdx, true);
}

double jetFlowSolver::misfit(){
	double cost;
	cost = 0.5*diff->vector()->inner(dJdx);

	std::shared_ptr<Matrix> Q = std::make_shared<Matrix>();
	std::shared_ptr<Matrix> Qt = std::make_shared<Matrix>(); 
	auto Qelement = std::make_shared<RANSCostW::Form_Qform_FunctionSpace_1>(geo->geoMesh);
	RANSCostW::Form_Qform Qform(Qelement, Qelement);	

	assemble(*Q, Qform);
	Q->mult(*(m->vector()) , dJdm);

	cost += 0.5*m->vector()->inner(dJdm);

	return cost;
};

void jetFlowSolver::grad(double gradientVector[]){
	auto Jm = std::make_shared<RANSPseudoTimeStepping::Form_Jm>(Q);

	Jm->m        = m;
	Jm->x        = x;  
	Jm->xl       = xl;
	Jm->ds       = geo->boundaryParts;
	Jm->u_ff     = u_ff;
	Jm->adjoints = adjoints;

	Vector Jm_vec;
	assemble(Jm_vec, *Jm);

	Jm_vec.axpy(1., dJdm);

	gradientVector[0] = Jm_vec.inner(*(subFun1->vector()));
	gradientVector[1] = Jm_vec.inner(*(subFun2->vector()));
};

std::shared_ptr<Matrix> jetFlowSolver::hessian(double hessianMatrixArray[]){
	//Set Linearization Point
	auto Rform = std::make_shared<RANSPseudoTimeStepping::Form_Rform>(Q, Q);
	auto Aform = std::make_shared<RANSPseudoTimeStepping::Form_Aform>(Vh_STATE, Vh_STATE);
	auto Cform = std::make_shared<RANSPseudoTimeStepping::Form_Cform>(Q, Vh_STATE);
	auto WuuForm = std::make_shared<RANSPseudoTimeStepping::Form_WuuForm>(Vh_STATE, Vh_STATE);

	//Compute R
	Matrix R;
	Rform->m        = m;
	Rform->x        = x;  
	Rform->xl       = xl;
	Rform->ds       = geo->boundaryParts;
	Rform->u_ff     = u_ff;
	Rform->adjoints = adjoints;
	assemble(R, *Rform);

	//Compute C
	Matrix C;
	Cform->m        = m;
	Cform->x        = x;  
	Cform->xl       = xl;
	Cform->ds       = geo->boundaryParts;
	Cform->u_ff     = u_ff;

	assemble(C, *Cform);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->zero(C);
	}

	//Compute Ct
	std::shared_ptr<Matrix> Ct = std::make_shared<Matrix>(); 
	Ct = Transpose(C);

	//Compute A-1
	Matrix A;
	Aform->sigma    = sigma;
	Aform->m        = m;
	Aform->x        = x;  
	Aform->xl       = xl;
	Aform->ds       = geo->boundaryParts;

	assemble(A, *Aform);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->apply(A);
	}

	// //Compute A-t
	// std::shared_ptr<Matrix> At = std::make_shared<Matrix>(); 
	// At = Transpose(A);

	//Compute Wuu
	Matrix Wuu;
	WuuForm->m        = m;
	WuuForm->x        = x;  
	WuuForm->xl       = xl;
	WuuForm->ds       = geo->boundaryParts;
	WuuForm->adjoints = adjoints;	

	assemble(Wuu, *WuuForm);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->zero(Wuu);
	}

	Vector workSpaceA; Vector workSpaceB;
	auto stateWorkSpace = std::make_shared<Function>(Vh_STATE);
	double term1; double term2; double singleEntry;
	PETScLUSolver solver;
	PETScMatrix Ap = as_type<const PETScMatrix>(A);
	solver.set_operator(Ap);


	Mat hessianMatrix;
    MatCreate(PETSC_COMM_SELF,&hessianMatrix);
    MatSetType(hessianMatrix, MATSEQDENSE);
    MatSetSizes(hessianMatrix,PETSC_DECIDE,PETSC_DECIDE,2,2);
    MatSetFromOptions(hessianMatrix);
    MatSetUp(hessianMatrix);
	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 2; j++){

			R.mult(*(subFunctions[i]->vector()), workSpaceA);
			term1 = workSpaceA.inner(*(subFunctions[j]->vector()));

			C.mult(*(subFunctions[i]->vector()), workSpaceB);
			solver.solve(*(stateWorkSpace->vector()), workSpaceB, false);
			Wuu.mult(*(stateWorkSpace->vector()), workSpaceB);
			solver.solve(*(stateWorkSpace->vector()), workSpaceB, true);
			Ct->mult(*(stateWorkSpace->vector()), workSpaceA);
			term2 = workSpaceA.inner(*(subFunctions[j]->vector()));

			singleEntry = term1 + term2;
			MatSetValue(hessianMatrix, j, i, singleEntry, INSERT_VALUES);
			hessianMatrixArray[j*2+i] = singleEntry;
		}
	}
	MatAssemblyBegin(hessianMatrix, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(hessianMatrix, MAT_FINAL_ASSEMBLY);

	std::shared_ptr<Matrix> out(new Matrix(PETScMatrix(hessianMatrix)));
	MatDestroy(&hessianMatrix);

	return out;
}


std::shared_ptr<Matrix> jetFlowSolver::Transpose(const GenericMatrix & A){
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


