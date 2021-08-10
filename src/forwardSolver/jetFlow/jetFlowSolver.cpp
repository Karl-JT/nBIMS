#include "jetFlowSolver.h"

jetFlowGeo::jetFlowGeo(int nx_, int ny_): nx(nx_), ny(ny_), P0(2, point0), P1(2, point1){
	geoMesh = std::make_shared<RectangleMesh>(P0, P1, nx_, ny_);
	comm = geoMesh->mpi_comm();
	rank = MPI::rank(comm);
	nprocs = MPI::size(comm);

	boundaryParts = std::make_shared<MeshFunction<size_t>>(geoMesh, geoMesh->topology().dim()-1);
	boundaryParts->set_all(0);

	InletBoundary Gamma_inlet;
	SymmetryBoundary Gamma_axis;
	FarfieldBoundary Gamma_farfield;
	upperBoundary Gamma_upper;

	Gamma_inlet.mark(*boundaryParts, 1);
	Gamma_axis.mark(*boundaryParts, 2);
	Gamma_farfield.mark(*boundaryParts, 3);
	Gamma_upper.mark(*boundaryParts, 4);

	domainParts = std::make_shared<MeshFunction<size_t>>(geoMesh, geoMesh->topology().dim());
	domainParts->set_all(0);

	InnerDomain Beta_inner;
	Beta_inner.mark(*domainParts, INNER);
}

void jetFlowGeo::meshScale(){
	auto coord = geoMesh->coordinates();	
	for (unsigned i = 0; i < coord.size(); ++i){
		if (i % 2 != 0){
			coord[i] = 10.0-10.0*tanh((1.0-coord[i]/10.0)*1.5)/tanh(1.5);
		}
	}
	geoMesh->bounding_box_tree()->build(*geoMesh.get());
}

bool fileExists(const std::string& file) {
    struct stat buf;
    return (stat(file.c_str(), &buf) == 0);
}

jetFlowSolver::jetFlowSolver(int nx_, int ny_, int level_) : level(level_){
	set_log_level(30);
	geo = std::make_shared<jetFlowGeo>(nx_, ny_);
	bbt = geo->geoMesh->bounding_box_tree();

	Vh_STATE = std::make_shared<RANSPseudoTimeStepping::CoefficientSpace_x>(geo->geoMesh);
	Q = std::make_shared<RANSPseudoTimeStepping::CoefficientSpace_m>(geo->geoMesh);

	m = std::make_shared<Function>(Vh_STATE->sub(2)->collapse());
	m->interpolate(mExpression);

	F = std::make_shared<RANSPseudoTimeStepping::Form_F>(Vh_STATE);

	J = std::make_shared<RANSPseudoTimeStepping::Form_J>(Vh_STATE, Vh_STATE);
	J_true = std::make_shared<RANSPseudoTimeStepping::Form_J_true>(Vh_STATE, Vh_STATE);


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
	
	u = std::make_shared<Function>(Vh_STATE->sub(0)->collapse());
	p = std::make_shared<Function>(Vh_STATE->sub(1)->collapse());
	k = std::make_shared<Function>(Vh_STATE->sub(2)->collapse());
	e = std::make_shared<Function>(Vh_STATE->sub(3)->collapse());

	ux = std::make_shared<Function>(Vh_STATE->sub(0)->sub(0)->collapse());
	uy = std::make_shared<Function>(Vh_STATE->sub(0)->sub(1)->collapse());

	d  = std::make_shared<Function>(Vh_STATE->sub(0)->sub(0)->collapse());
	xd = std::make_shared<Function>(Vh_STATE);
	diff = std::make_shared<Function>(Vh_STATE);

	//Compute Trans
	mTran1 subFun1_expression;
	mTran2 subFun2_expression;
	subFun1 = std::make_shared<Function>(Q);
	subFun2 = std::make_shared<Function>(Q);
	subFun1->interpolate(subFun1_expression);
	subFun2->interpolate(subFun2_expression);
	subFunctions = {subFun1, subFun2};	
};

void jetFlowSolver::soluFwd(){
	if (geo->rank == 0){
		std::cout << "start forward solution" << std::endl;
	}

	data_file += std::to_string(level);
	data_file += ".h5";
	std::cout << data_file << std::endl;
	if (fileExists(data_file)){
		std::cout << "reading input file" << std::endl;
		auto inputMesh = std::make_shared<Mesh>();
		HDF5File inputFile(geo->comm, data_file, "r");
		inputFile.read(*inputMesh.get(), "/mesh", false);
		auto inputState = std::make_shared<RANSPseudoTimeStepping::CoefficientSpace_x>(inputMesh);
		auto inputData = std::make_shared<Function>(inputState);
		inputFile.read(*inputData.get(), "/states");
		LagrangeInterpolator::interpolate(*x.get(), *inputData.get());
		assign(xl, x);
		inputFile.close();
	} else {
		initCond zeroInit;
		x->interpolate(zeroInit);
		assign(xl, x);
	}

	m->interpolate(mExpression);
	double sigmaUpdate = 200.0;
	sigma.reset(new Constant(sigmaUpdate)); 

	F->sigma = sigma;
	F->m     = m;
	F->x     = x;  
	F->xl    = xl;
	F->ds    = geo->boundaryParts;

	J->sigma = sigma;
	J->m    = m;
	J->x    = x;
	J->xl   = xl;
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
	double r_norm_p;

	solve(*F == 0, *x, bcs, *J, params);

	assign(xl, x);
	F->sigma = sigma;
	F->x  = x; F->xl = xl;
	J->sigma = sigma;
	J->x  = x; J->xl = xl;	
	assemble(r, *F);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->apply(r);
	}
	r_norm = norm(r, "l2");
	r_norm_p = 100;

	if (geo->rank == 0){
	printf("norm: %e", r_norm);
	}

	for (int i = 0; i < 100000; i++){
		assign(xl, x);
		F->sigma = sigma;
		F->x  = x; F->xl = xl;
		J->sigma = sigma;
		J->x  = x; J->xl = xl;

		assemble(r, *F);
		for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
			bcs0[bcCount]->apply(r);
		}
		r_norm = norm(r, "l2");
	 	solve(*F == 0, *x, bcs, *J, params);

	 	if (r_norm_p == 0.001){ 
	 		break;
	 	} 

 		sigmaUpdate = std::max(0.001, 0.5*(r_norm_p + r_norm*100));
 		r_norm_p = sigmaUpdate;
		sigma.reset(new Constant(sigmaUpdate)); 

		if (geo->rank == 0){
		printf("\rnorm: %e                ", r_norm);
		}
	 	updateInitialState();
	}
	if (geo->rank == 0){
		std::cout << std::endl << "forward solve completed" << std::endl;
	}
};

void jetFlowSolver::updateInitialState(){
	assign(xl, x);
	File uFile("u.pvd");
	File kFile("k.pvd");
	File pFile("p.pvd");
	File eFile("e.pvd");
	assign({u, p, k, e}, x);

	uFile << *u;
	kFile << *k;
	eFile << *e;
	pFile << *p;

	HDF5File outputFile(geo->comm, data_file, "w");
	outputFile.write(*geo->geoMesh.get(), "mesh"); 
	outputFile.write(*x.get(), "states");
	outputFile.close();
}

void jetFlowSolver::updateParameter(double parameter[], int numCoef){

	mExpression.coef[0] = parameter[0];
	mExpression.coef[1] = parameter[1];

	if (geo->rank == 0){
		std::cout << "update control parameter to: " << parameter[0] << " " << parameter[1] << std::endl;
	}
};

void jetFlowSolver::generateRealization(){
	if (geo->rank == 0){
		std::cout << "generate realization" << std::endl;
	}

	double noise_level = 0.001;
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, noise_level);

	assign({ux, uy}, u);
	assign(d, ux);

	Vector dNoise;
	dNoise.init(d->vector()->size());
	for (unsigned i = 0; i < d->vector()->size(); i++){
		dNoise.setitem(i, distribution(generator));
	}
	d->vector()->axpy(1, dNoise);	//Not distributed, Error during mpirun!!!

	File dFile("d.pvd");
	dFile << *d;
};

void jetFlowSolver::generatePointRealization(){
	if (geo->rank == 0){
		std::cout << "generate point realization" << std::endl;
	}

	int num = 15*5;
	double noise_level = 0.001;

	for (int i = 0; i < 15; i++){
		for (int j = 0; j < 5; j++){
			obsPoints[i*10+2*j]   = i/0.5 + 1.0/1;
			obsPoints[i*10+2*j+1] = j/0.5 + 1.0/1;
		}
	}

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, noise_level);

	assign({ux, uy}, u);

	std::cout << "function interpolation" << std::endl;

	for (int i = 0; i < num; ++i){
		Point thisPoint(2, obsPoints+2*i);
		if (bbt->collides_entity(thisPoint)){
			obsValues[i] = (*ux)(obsPoints[2*i], obsPoints[2*i+1]);
			obsValues[i] += distribution(generator);
		} else {
			obsValues[i] = 1e16;
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, obsValues, num, MPI_DOUBLE, MPI_MIN, geo->comm);

	if (geo->rank == 0){
		std::string pointFile = "obsPoints.txt";
		std::string valueFile = "obsValues.txt";
		write2txt(obsPoints, 2*num, pointFile);
		write2txt(obsValues, num, valueFile);
	}
};


void jetFlowSolver::soluAdj(){
	// std::string data_name = "adjoints";
	// std::string data_file = "adjoints.h5";
	// if (fileExists(data_file)){
	// 	std::cout << "reading input file" << std::endl;
	// 	auto inputMesh = std::make_shared<Mesh>();
	// 	HDF5File inputFile(MPI_COMM_WORLD, "adjoints.h5", "r");
	// 	inputFile.read(*inputMesh.get(), "/mesh", false);
	// 	auto inputState = std::make_shared<RANSPseudoTimeStepping::CoefficientSpace_x>(inputMesh);
	// 	auto inputData = std::make_shared<Function>(inputState);
	// 	inputFile.read(*inputData.get(), "/adjoints");
	// 	LagrangeInterpolator::interpolate(*adjoints.get(), *inputData.get());
	// 	inputFile.close();
	// } else {
	// 	initCond zeroInit;
	// 	adjoints->interpolate(zeroInit);
	// }

	sigma.reset(new Constant(0.0));
	PETScKrylovSolver solver("gmres", "jacobi");
	solver.parameters.remove("relative_tolerance");
	solver.parameters.add("relative_tolerance", 1e-4);
	// solver.parameters.remove("monitor_convergence");
	// solver.parameters.add("monitor_convergence", true);
	solver.parameters.remove("maximum_iterations");
	solver.parameters.add("maximum_iterations", 90000);
	solver.parameters.add("restart", 300);
	solver.parameters.remove("error_on_nonconvergence");
	solver.parameters.add("error_on_nonconvergence", false);
	// solver.parameters.remove("nonzero_initial_guess");
	// solver.parameters.add("nonzero_initial_guess", false);

	J->sigma = sigma;
	J->m     = m;
	J->x     = x;
	J->xl    = xl;
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

	dJdx.str(true);
	solver.solve(*(adjoints->vector()), dJdx, true);
	// adjoints->vector()->str(true);

	// HDF5File outputFile(geo->comm, "adjoints.h5", "w");
	// outputFile.write(*geo->geoMesh.get(), "mesh"); 
	// outputFile.write(*adjoints.get(), "adjoints");
	// outputFile.close();
}

double jetFlowSolver::pointwiseMisfit(){
	if (geo->rank == 0){
		std::cout << "compute point misfit" << std::endl;
	}

	double misfit = 0;
	// double cost;

	std::string pointFile = "obsPoints.txt";
	std::string valueFile = "obsValues.txt";
	txt2read(obsPoints, 150, pointFile);
	txt2read(obsValues, 75, valueFile);
	
	pointwiseTransform(obsPoints, 75);
	std::shared_ptr<const GenericDofMap> dofmap = Vh_STATE->dofmap();

	Vector temp;
	pointwiseTransformMatrix->mult(*(x->vector()), temp);

	// if (geo->rank == 0){
	// 	double statesVector[x->vector()->size()];
	// 	la_index rowIdx[x->vector()->size()];
	// 	for (unsigned i = 0; i < x->vector()->size(); i++){
	// 		rowIdx[i] = i;
	// 	}
	// 	x->vector()->get(statesVector, x->vector()->size(), rowIdx);
	// 	for (unsigned i = 0; i < x->vector()->size(); i++){
	// 		std::cout << i << " " << statesVector[i] << " " << std::endl;
	// 	}		
	// }

	// x->vector()->str(true);


	Vec obsTemp;
	VecCreate(geo->comm, &obsTemp);
	VecSetSizes(obsTemp, PETSC_DECIDE, temp.size());
	VecSetFromOptions(obsTemp);

	std::cout << "local_obsNum: " << local_obsNum << std::endl;

	std::cout << "misfit compute" << std::endl;
	for (int i = 0; i < local_obsNum; ++i){
		// if (i%5 == 0){

		if (geo->rank == 1){
			std::cout << "local index: " << i << " global index: " <<  LG[i] <<  " " << temp[LG[i]*5] << " " << obsValues[LG[i]] << std::endl;
			std::cout << temp[LG[i]*5+1] << " " << temp[LG[i]*5+2] << " " << temp[LG[i]*5+3] << " " << temp[LG[i]*5+4] << std::endl;
		}
		VecSetValue(obsTemp, LG[i]*5, 1e6*(temp[LG[i]*5]-obsValues[LG[i]]), INSERT_VALUES);
		VecSetValue(obsTemp, LG[i]*5+1, 0, INSERT_VALUES);
		VecSetValue(obsTemp, LG[i]*5+2, 0, INSERT_VALUES);
		VecSetValue(obsTemp, LG[i]*5+3, 0, INSERT_VALUES);
		VecSetValue(obsTemp, LG[i]*5+4, 0, INSERT_VALUES);
		// } 
		// else {
		// 	VecSetValue(obsTemp, i, 0, INSERT_VALUES);			
		// }
	}
	MPI_Barrier(geo->comm);
	VecAssemblyBegin(obsTemp);VecAssemblyEnd(obsTemp);

	auto temp2 = std::make_shared<Vector>(PETScVector(obsTemp));
	pointwiseTransformMatrix->transpmult(*temp2, dJdx);

	// if (geo->rank == 0){
	// 	double statesVector[x->vector()->size()];
	// 	la_index rowIdx[x->vector()->size()];
	// 	for (unsigned i = 0; i < x->vector()->size(); i++){
	// 		rowIdx[i] = i;
	// 	}
	// 	x->vector()->get(statesVector, x->vector()->size(), rowIdx);
	// 	for (unsigned i = 0; i < x->vector()->size(); i++){
	// 		std::cout << i << " " << statesVector[i] << " " << std::endl;
	// 	}		
	// }

	assign({ux, uy}, u);

	for (int i = 0; i < 75; ++i){
		Point thisPoint(2, obsPoints+2*i);
		if (bbt->collides_entity(thisPoint)){
			misfit += 0.5*1e6*pow((*ux)(obsPoints[2*i], obsPoints[2*i+1])-obsValues[i], 2); //Does not work in parallel
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, &misfit, 1, MPI_DOUBLE, MPI_SUM, geo->comm);

	// cost = misfit + 0.5*pow(mExpression.coef[0], 2) + pow(mExpression.coef[1], 2);
	if (geo->rank == 0){
		std::cout << "misfit: " << misfit << std::endl;
	}
	// std::cout << "cost" << cost << std::endl;
	return misfit;
}

double jetFlowSolver::misfit(){

	std::shared_ptr<Matrix> W = std::make_shared<Matrix>();
	std::shared_ptr<Matrix> Wt = std::make_shared<Matrix>(); 
	RANSPseudoTimeStepping::Form_WuuFormAdd Wform(Vh_STATE, Vh_STATE);	

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
	assign({u, p, k, e}, x);
	assign({ux, uy}, u);
	assign(u, {d, uy});
	assign(xd, {u, p, k, e});

	diff->vector()->axpy(-1, *(xd->vector()));
	W->mult(*(diff->vector()) , dJdx);
	dJdx *= 1.0e6;

	double cost;
	cost = -0.5*diff->vector()->inner(dJdx);

	std::shared_ptr<Matrix> Q = std::make_shared<Matrix>();
	std::shared_ptr<Matrix> Qt = std::make_shared<Matrix>(); 
	auto Qelement = std::make_shared<RANSCostW::Form_Qform_FunctionSpace_1>(geo->geoMesh);
	RANSCostW::Form_Qform Qform(Qelement, Qelement);	

	assemble(*Q, Qform);
	Q->mult(*(m->vector()) , dJdm);

	// cost -= 0.5*pow(mExpression.coef[0], 2);
	// cost -= 0.5*pow(mExpression.coef[1], 2);

	std::cout << "cost: " << cost << std::endl;
	return cost;
};

int jetFlowSolver::pointwiseTransform(double obsPoints[], int obsNum){
	// const Mesh& mesh = *(Vh_STATE->sub(0)->mesh());
	// const int num_cells = mesh.num_cells();
	points.clear();
	LG.clear();
	points.reserve(obsNum);
	LG.reserve(obsNum);

	std::vector<int> tmp(obsNum);
	double *p = obsPoints;
	for (int i = 0; i < obsNum; ++i){
		Point thisPoint(2, p+2*i);
		if (bbt->collides_entity(thisPoint)){
			tmp[i] = geo->rank;
		} else {
			tmp[i] = geo->nprocs;
		}
	}

	std::vector<int> owner(obsNum);
	MPI_Allreduce(tmp.data(), owner.data(), obsNum, MPI_INT, MPI_MIN, geo->comm);

	for (int i = 0; i < obsNum; ++i){
		if (owner[i] == geo->rank){
			LG.push_back(i);
			points.push_back( Point(2, p+2*i));
		}
	}

	// std::vector<PetscInt> proc_offset(geo->nprocs+1);
	// std::fill(proc_offset.begin(), proc_offset.end(), 0);
	// for(int i = 0; i < obsNum; ++i){
	// 	++proc_offset[owner[i]+1];
	// }
	// std::partial_sum(proc_offset.begin(), proc_offset.end(), proc_offset.begin());
	// std::vector<double> old_new(obsNum);
	// for(int i = 0; i < obsNum; ++i)
	// {
	// 	old_new[i] = proc_offset[owner[i]];
	// 	++proc_offset[owner[i]];
	// }
	// for(std::size_t jj = 0; jj < LG.size(); ++jj){
	// 	LG[jj] = old_new[LG[jj]];
	// }

	global_obsNum = obsNum;
	local_obsNum = points.size();

	std::shared_ptr<const FiniteElement> element( Vh_STATE->element() );
	int value_dim = element->value_dimension(0);

	std::shared_ptr<const GenericDofMap> dofmap = Vh_STATE->dofmap();
	PetscInt global_dof_dimension = dofmap->global_dimension();
	PetscInt local_dof_dimension = dofmap->index_map()->size(IndexMap::MapSize::OWNED);

	/////////////////////////////////////////////

	PetscInt global_nrows = global_obsNum*value_dim;
	PetscInt local_nrows = local_obsNum*value_dim;

	std::vector<la_index> LGdofs = dofmap->dofs();
	std::vector<PetscInt> LGrows(local_nrows);

	int counter = 0;
	for (int lt = 0; lt < local_obsNum; ++lt){
		for (int ival = 0; ival < value_dim; ++ival, ++counter){
			LGrows[counter] = LG[lt]*value_dim + ival;
		}
	}

	Mat pObsMat;
	MatCreate(geo->comm, &pObsMat); 
	MatSetSizes(pObsMat, local_nrows, local_dof_dimension, global_nrows, global_dof_dimension);
	MatSetType(pObsMat, MATAIJ);
	MatSetUp(pObsMat);

	ISLocalToGlobalMapping rmapping, cmapping;
	PetscCopyMode mode = PETSC_COPY_VALUES;

	PetscInt bs = 1;
	ISLocalToGlobalMappingCreate(geo->comm, bs, LGrows.size(), &LGrows[0], mode, &rmapping);
	ISLocalToGlobalMappingCreate(geo->comm, bs, LGdofs.size(), &LGdofs[0], mode, &cmapping);

	MatSetLocalToGlobalMapping(pObsMat, rmapping, cmapping);

	std::size_t sdim = element->space_dimension();
	std::vector<double> basis_matrix(sdim*value_dim);
	std::vector<double> basis_matrix_row_major(sdim*value_dim);
	std::vector<PetscInt> cols(sdim);

	for (int gi = 0; gi < local_obsNum; ++gi){

		if (geo->rank == 1){
			std::cout << "local index: " << gi << " global index: " <<  LG[gi] << std::endl;
		}
		//////////////////////No issue/////////////////////
		int cell_id = bbt->compute_first_entity_collision(points[gi]);
		Cell cell(*(geo->geoMesh), cell_id);
		std::vector<double> coords;
		// cell.get_vertex_coordinates(coords);
		cell.get_coordinate_dofs(coords);
		element->evaluate_basis_all(&basis_matrix[0], points[gi].coordinates(), &coords[0], cell.orientation());
		///////////////////////////////////////////////////

		Eigen::Map<const Eigen::Array<la_index, Eigen::Dynamic, 1>> cell_dofs = dofmap->cell_dofs(cell_id);

		auto it_col = cols.begin();
		for (std::size_t it = 0; it < sdim; ++it, ++it_col){
			*it_col = dofmap->local_to_global_index(cell_dofs[it]);
		} 
		for (std::size_t i = 0; i < sdim; ++i){
			for (int j = 0; j < value_dim; j++){
				basis_matrix_row_major[i+j*sdim] = basis_matrix[value_dim*i+j];
			}
		}

		// if (geo->rank == 0){
		// 	std::cout << "point coordinates: " << points[gi].coordinates()[0] << " " << points[gi].coordinates()[1] << std::endl;
		// 	std::cout << std::endl;
		// 	for (auto j = cols.begin(); j != cols.end(); ++j){
		// 		std::cout << *j << " ";
		// 	}
		// 	std::cout << std::endl;
		// }
		MatSetValues(pObsMat, value_dim, &LGrows[value_dim*gi], sdim, &cols[0], &basis_matrix_row_major[0], INSERT_VALUES);
	}

	MPI_Barrier(geo->comm);
	MatAssemblyBegin(pObsMat, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(pObsMat, MAT_FINAL_ASSEMBLY);

	// MatView(pObsMat, PETSC_VIEWER_STDOUT_WORLD);

	pointwiseTransformMatrix = std::make_shared<Matrix>(PETScMatrix(pObsMat));
	// pointwiseTransformMatrix->str(true);

	MatDestroy(&pObsMat);

	return 0;
};

void jetFlowSolver::grad(double gradientVector[]){

	auto Jm = std::make_shared<RANSPseudoTimeStepping::Form_Jm>(Q);

	Jm->m        = m; 
	Jm->x        = x;  
	Jm->xl       = xl;
	Jm->ds       = geo->boundaryParts;
	Jm->adjoints = adjoints;


	assemble(Jm_vec, *Jm);

	// std::shared_ptr<Matrix> Q = std::make_shared<Matrix>();
	// std::shared_ptr<Matrix> Qt = std::make_shared<Matrix>(); 
	// auto Qelement = std::make_shared<RANSCostW::Form_Qform_FunctionSpace_1>(geo->geoMesh);
	// RANSCostW::Form_Qform Qform(Qelement, Qelement);	

	// assemble(*Q, Qform);
	// Q->mult(*(m->vector()) , dJdm);
	// Jm_vec.axpy(-1.0, dJdm);

	gradientVector[0] = -Jm_vec.inner(*(subFunctions[0]->vector()));
	gradientVector[1] = -Jm_vec.inner(*(subFunctions[1]->vector()));

	// gradientVector[0] += 2*mExpression.coef[0];
	// gradientVector[1] += 2*mExpression.coef[1];

	// std::cout << "Jm_vec" << std::endl;
	// Jm_vec.str(true);

	std::cout << "gradient vector: " << gradientVector[0] << " " << gradientVector[1] << std::endl;
};

std::shared_ptr<Matrix> jetFlowSolver::hessian(double hessianMatrixArray[]){
	sigma.reset(new Constant(0.0));
	//Set Linearization Point
	// auto Rform = std::make_shared<RANSPseudoTimeStepping::Form_Rform>(Q, Q);
	auto Aform = std::make_shared<RANSPseudoTimeStepping::Form_Aform>(Vh_STATE, Vh_STATE);
	auto Cform = std::make_shared<RANSPseudoTimeStepping::Form_Cform>(Q, Vh_STATE);
	// auto WuuForm = std::make_shared<RANSPseudoTimeStepping::Form_WuuForm>(Vh_STATE, Vh_STATE);
	// auto WauForm = std::make_shared<RANSPseudoTimeStepping::Form_WauForm>(Vh_STATE, Q);
	// auto WuaForm = std::make_shared<RANSPseudoTimeStepping::Form_WuaForm>(Q, Vh_STATE);
	auto WuuFormAdd = std::make_shared<RANSPseudoTimeStepping::Form_WuuFormAdd>(Vh_STATE, Vh_STATE);
	// auto RFormAdd = std::make_shared<RANSPseudoTimeStepping::Form_RFormAdd>(Q, Q);

	auto Jx = std::make_shared<RANSPseudoTimeStepping::Form_Jx>(Vh_STATE);
	auto Jadj = std::make_shared<RANSPseudoTimeStepping::Form_Jadj>(Vh_STATE);

	//Compute R
	// Matrix R;
	// Rform->m        = m;
	// Rform->x        = x;  
	// Rform->xl       = xl;
	// Rform->ds       = geo->boundaryParts;
	// Rform->adjoints = adjoints;
	// assemble(R, *Rform);

	// auto RAdd = std::make_shared<Matrix>(); 
	// auto RAddT = std::make_shared<Matrix>(); 
	// RFormAdd->ds = geo->boundaryParts;

	// assemble(*RAdd, *RFormAdd);
	// WuuFinal->str(true);

	// R.axpy(1.0, *RAdd, false);


	//Compute C
	Matrix C;
	Cform->m        = m;
	Cform->x        = x;  
	Cform->xl       = xl;
	Cform->ds       = geo->boundaryParts;

	assemble(C, *Cform);
	for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
		bcs0[bcCount]->zero(C);
	}

	PETScMatrix Cp = as_type<const PETScMatrix>(C);


	//Compute Ct
	// std::shared_ptr<Matrix> Ct = std::make_shared<Matrix>(); 
	// Ct = Transpose(C);

	// Matrix Wua;
	// WuaForm->m        = m;
	// WuaForm->x        = x;  
	// WuaForm->xl       = xl;
	// WuaForm->adjoints = adjoints;
	// WuaForm->ds       = geo->boundaryParts;
	// assemble(Wua, *WuaForm);
	// for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
	// 	bcs0[bcCount]->zero(Wua);
	// }


	// Matrix Wau;
	// WauForm->m        = m;
	// WauForm->x        = x;  
	// WauForm->xl       = xl;
	// WauForm->adjoints = adjoints;
	// WauForm->ds       = geo->boundaryParts;
	// assemble(Wau, *WauForm);
	// for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
	// 	bcs0[bcCount]->zero(Wau);
	// }	

	//Compute A-1
	Matrix A;
	Aform->sigma    = sigma;
	Aform->m        = m;
	Aform->x        = x;  
	Aform->xl       = xl;
	Aform->ds       = geo->boundaryParts;

	Jadj->sigma     = sigma;
	Jadj->x         = x;
	Jadj->xl        = xl;
	Jadj->m         = m;
	Jadj->ds        = geo->boundaryParts;

	// assemble(A, *Aform);
	Vector dummy;
	std::vector<std::shared_ptr<const DirichletBC>> bcs0p;
	bcs0p.push_back(u_homogenize); 
	bcs0p.push_back(k_homogenize); 
	bcs0p.push_back(e_homogenize); 
	bcs0p.push_back(axis_homogenize);
	assemble_system(A, dummy, *Aform, *Jadj, bcs0p);
	// for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
	// 	bcs0[bcCount]->apply(A);
	// }
	// A.str(true);

	// //Compute A-t
	// std::shared_ptr<Matrix> At = std::make_shared<Matrix>(); 
	// At = Transpose(A);

	//Compute Wuu
	// std::shared_ptr<Matrix> Wuu = std::make_shared<Matrix>(); 
	// Matrix Wuu;
	// WuuForm->m        = m;
	// WuuForm->x        = x;  
	// WuuForm->xl       = xl;
	// WuuForm->adjoints = adjoints;
	// WuuForm->ds       = geo->boundaryParts;

	// Jx->x         = x;
	// Jx->xl        = xl;
	// Jx->sigma     = sigma;
	// Jx->adjoints  = adjoints;
	// Jx->m         = m;
	// Jx->ds        = geo->boundaryParts;

	// assemble_system(Wuu, dummy, *WuuForm, *Jx, bcs0p);
	// for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
	// 	bcs0[bcCount]->zero(Wuu);
	// }

	// auto WuuAdd = std::make_shared<Matrix>(*pointwiseTransformMatrix); 




	//////////////////////////////////////////
	///// continuous function observation   //
	//////////////////////////////////////////
	// auto WuuFinal = std::make_shared<Matrix>(); 
	// auto WuuAddT = std::make_shared<Matrix>(); 
	// WuuFormAdd->ds = geo->boundaryParts;

	// assemble(*WuuFinal, *WuuFormAdd);
	// for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
	// 	bcs0[bcCount]->zero(*WuuFinal);
	// }
	// WuuAddT = dolfinTranspose(*WuuFinal);
	// for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
	// 	bcs0[bcCount]->zero(*WuuAddT);
	// }	
	// WuuFinal = dolfinTranspose(*WuuAddT);
	// (*WuuFinal) *= 1.0e6;


	///////////////////////////////////////
	//// point observation               //
	///////////////////////////////////////
	// auto WuuFinal = std::make_shared<Matrix>(); 
	auto WuuFinal = dolfinMatTransposeMatMult(*pointwiseTransformMatrix, *pointwiseTransformMatrix);
	(*WuuFinal) *= 1.0e6;

	// WuuFinal->axpy(-1.0, Wuu, false);

	// std::shared_ptr<Matrix> WuuT = std::make_shared<Matrix>(); 
    // WuuT = Transpose(*Wuu);
	// for (std::size_t bcCount = 0; bcCount < bcs0.size(); bcCount++){
	// 	bcs0[bcCount]->zero(*WuuT);
	// }
    // Wuu = Transpose(*WuuT);

	// Wuu.str(true);
	// PETScMatrix Wuup = as_type<const PETScMatrix>(Wuu);

	// PetscViewer viewer5;
	// PetscViewerBinaryOpen(geo->comm, "Wuu", FILE_MODE_WRITE, &viewer5);
	// MatView(Wuup.mat(),viewer5);
	// PetscViewerDestroy(&viewer5);

	Vector workSpaceA; Vector workSpaceB;
	auto stateWorkSpace = std::make_shared<Function>(Vh_STATE);
	double term2; double singleEntry; //double term1; 
	PETScKrylovSolver Asolver("gmres", "jacobi");
	Asolver.parameters.remove("relative_tolerance");
	Asolver.parameters.add("relative_tolerance", 1e-4);
	Asolver.parameters.remove("monitor_convergence");
	Asolver.parameters.add("monitor_convergence", true);
	Asolver.parameters.remove("maximum_iterations");
	Asolver.parameters.add("maximum_iterations", 90000);
	Asolver.parameters.add("restart", 300);
	Asolver.parameters.remove("error_on_nonconvergence");
	Asolver.parameters.add("error_on_nonconvergence", false);
	PETScMatrix Ap = as_type<const PETScMatrix>(A);
	// PETScMatrix reg(Ap);
	// Vector dig(geo->comm, Ap.size(0));
	// reg.zero(); reg.get_diagonal(dig); dig = 1; reg.set_diagonal(dig);
	// Ap.axpy(0.001, reg, false);
	Asolver.set_operators(Ap, Ap);

	// Hessian Matrix with Prior spectral
	Mat hessianMatrix;
    MatCreate(PETSC_COMM_SELF,&hessianMatrix);
    MatSetType(hessianMatrix, MATSEQDENSE);
    MatSetSizes(hessianMatrix,PETSC_DECIDE,PETSC_DECIDE,2,2);
    MatSetFromOptions(hessianMatrix);
    MatSetUp(hessianMatrix);
	for (int i = 0; i < 2; ++i){
		for (int j = 0; j < 2; ++j){ //i+1; ++j){

			// R.mult(*(subFunctions[j]->vector()), workSpaceA);
			// term1 = workSpaceA.inner(*(subFunctions[i]->vector()));
			// if (i == 0 && j == 0) {subFunctions[j]->vector()->str(true);}

			C.mult(*(subFunctions[j]->vector()), workSpaceB);
			Asolver.solve(*(stateWorkSpace->vector()), workSpaceB, false);
			std::cout << "Asolver passed" << std::endl;
			WuuFinal->mult(*(stateWorkSpace->vector()), workSpaceB);
			Asolver.solve(*(stateWorkSpace->vector()), workSpaceB, true);
			std::cout << "Asolver transpose passed" << std::endl;
			C.transpmult(*(stateWorkSpace->vector()), workSpaceA);
			term2 = workSpaceA.inner(*(subFunctions[i]->vector()));

			// Wua.mult(*(subFunctions[j]->vector()), workSpaceB);
			// Asolver.solve(*(stateWorkSpace->vector()), workSpaceB, true);
			// std::cout << "Asolver transpose passed" << std::endl;
			// C.transpmult(*(stateWorkSpace->vector()), workSpaceA);
			// term2 += workSpaceA.inner(*(subFunctions[i]->vector()));

			// C.mult(*(subFunctions[j]->vector()), workSpaceB);
			// Asolver.solve(*(stateWorkSpace->vector()), workSpaceB, false);
			// std::cout << "Asolver passed" << std::endl;
			// Wua.transpmult(*(stateWorkSpace->vector()), workSpaceA);
			// term2 += workSpaceA.inner(*(subFunctions[i]->vector()));		

			singleEntry = term2; // + term1;
			MatSetValue(hessianMatrix, i, j, -singleEntry, INSERT_VALUES);
			hessianMatrixArray[2*i+j] = -singleEntry;
			std::cout << i << " " << j << " " << singleEntry;
			// if (i != j){
			// 	MatSetValue(hessianMatrix, j, i, -singleEntry, INSERT_VALUES);
			// 	hessianMatrixArray[2*j+i] = -singleEntry;		
			// }
		}
	}
	MatAssemblyBegin(hessianMatrix, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(hessianMatrix, MAT_FINAL_ASSEMBLY);

	std::shared_ptr<Matrix> out(new Matrix(PETScMatrix(hessianMatrix)));
	out->str(true);

	/////////////////////////////////////////////////////////////////////////////////////////
	//
	//Randomized Algorithm for Hessian Approximation
	/////////////////////////////////////////////////////////////////////////////////////////
	// std::cout << "Randomized Algorithm for Approximation of Reduced Hessian" << std::endl;

	// //Compute hessian with randomize algorithm
	// std::default_random_engine generator;
	// std::normal_distribution<double> distribution(0.0, 1.0);

	// std::cout << "create random matrix" << std::endl;

	// Mat randomOmega;
	// MatCreate(geo->comm, &randomOmega);
	// MatSetSizes(randomOmega, PETSC_DECIDE, PETSC_DECIDE, m->vector()->size(), 15);
	// MatSetUp(randomOmega);
	// for (unsigned i = 0; i < m->vector()->size(); i++){
	// 	for (unsigned j = 0; j < 15; j++){
	// 		MatSetValue(randomOmega, i, j, distribution(generator), INSERT_VALUES);
	// 	}
	// }	
	// MatAssemblyBegin(randomOmega, MAT_FINAL_ASSEMBLY);
	// MatAssemblyEnd(randomOmega, MAT_FINAL_ASSEMBLY);

	// // PetscViewer viewer1;
	// // PetscViewerBinaryOpen(geo->comm, "randomOmega", FILE_MODE_WRITE, &viewer1);
	// // MatView(Ap.mat(),viewer1);
	// // PetscViewerDestroy(&viewer1);
	// // std::cout << "create Y matrix" << std::endl;

	// Mat yMat; Vec inputSpace;
	// VecCreate(geo->comm, &inputSpace);
	// VecSetSizes(inputSpace, PETSC_DECIDE, m->vector()->size());
	// VecSetFromOptions(inputSpace);
	// VecSetUp(inputSpace);

	// MatCreateSeqDense(geo->comm, m->vector()->size(), 15, NULL, &yMat);
	// for (int i = 0; i < 15; i++){
	// 	MatGetColumnVector(randomOmega, inputSpace, i);
 //    	PETScVector inputSpaceDolfin = PETScVector(inputSpace);

	// 	C.mult(inputSpaceDolfin, workSpaceB);
	// 	std::cout << "Asolver pass" << std::endl;
	// 	Asolver.solve(*(stateWorkSpace->vector()), workSpaceB, false);
	// 	WuuFinal->mult(*(stateWorkSpace->vector()), workSpaceB);	
	// 	std::cout << "Asolver transpose pass" << std::endl;	
	// 	Asolver.solve(*(stateWorkSpace->vector()), workSpaceB, true);
	// 	C.transpmult(*(stateWorkSpace->vector()), inputSpaceDolfin);

	// 	PetscScalar *outputSpace;
	// 	VecGetArray(inputSpaceDolfin.vec(), &outputSpace);
	// 	for (unsigned j = 0; j < m->vector()->size(); j++){
	// 		MatSetValue(yMat, j, i, outputSpace[j], INSERT_VALUES);
	// 	}
	// }
	// MatAssemblyBegin(yMat, MAT_FINAL_ASSEMBLY);
	// MatAssemblyEnd(yMat, MAT_FINAL_ASSEMBLY);

	// // std::cout << "yMat size" << std::endl;
	// // MatView(yMat, PETSC_VIEWER_STDOUT_WORLD);

	// std::cout << "QR decomposition" << std::endl;

	// BV bv;
	// Mat Q;
	// BVCreateFromMat(yMat, &bv);
	// BVSetFromOptions(bv);
	// BVOrthogonalize(bv, NULL);
	// BVCreateMat(bv, &Q);

	// std::cout << "initialized eigenvalue problem" << std::endl;

	// EPS eps;
	// Mat T, S, lambda, U, UTranspose, ReducedHessian;
	// Vec workSpace3, xr, xi;
	// PetscScalar kr, ki;
	// MatCreate(geo->comm, &T);
	// MatCreate(geo->comm, &S);
	// MatCreate(geo->comm, &U);
	// MatCreate(geo->comm, &lambda);
	// VecCreate(geo->comm, &workSpace3);
	// MatSetSizes(T, PETSC_DECIDE, PETSC_DECIDE, 15, 15);
	// MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, 15, 15);
	// MatSetSizes(U, PETSC_DECIDE, PETSC_DECIDE, 15, 15);
	// MatSetSizes(lambda, PETSC_DECIDE, PETSC_DECIDE, 15, 15);
	// VecSetSizes(workSpace3, PETSC_DECIDE, 15);
	// MatSetFromOptions(T);
	// MatSetFromOptions(S);
	// MatSetFromOptions(U);
	// MatSetFromOptions(lambda);
	// VecSetFromOptions(workSpace3);
	// MatSetUp(T);
	// MatSetUp(S);
	// MatSetUp(lambda);
	// MatSetUp(U);
	// VecSetUp(workSpace3);


	// std::cout << "Assemble T matrix" << std::endl;
	// for (int i = 0; i < 15; i++){
	// 	MatGetColumnVector(Q, inputSpace, i);
 //    	PETScVector inputSpaceDolfin = PETScVector(inputSpace);

 //    	// inputSpaceDolfin.str(true);
	// 	C.mult(inputSpaceDolfin, workSpaceB);
	// 	std::cout << "Asolver pass" << std::endl;
	// 	Asolver.solve(*(stateWorkSpace->vector()), workSpaceB, false);
	// 	WuuFinal->mult(*(stateWorkSpace->vector()), workSpaceB);		
	// 	std::cout << "AsolverTranspose pass" << std::endl;
	// 	// workSpaceB.str(true);
	// 	Asolver.solve(*(stateWorkSpace->vector()), workSpaceB, true);
	// 	C.transpmult(*(stateWorkSpace->vector()), inputSpaceDolfin);
	// 	std::cout << "final " << std::endl;
	// 	// inputSpaceDolfin.str(true);

	// 	MatMultTranspose(Q, inputSpaceDolfin.vec(), workSpace3);
	// 	PetscScalar *outputSpace;
	// 	VecGetArray(workSpace3, &outputSpace);
	// 	for (int j = 0; j < 15; j++){
	// 		MatSetValue(T, j, i, outputSpace[j], INSERT_VALUES);
	// 	}
	// }
	// MatAssemblyBegin(T, MAT_FINAL_ASSEMBLY);
	// MatAssemblyEnd(T, MAT_FINAL_ASSEMBLY);
	// MatCreateVecs(T, NULL, &xr);
	// MatCreateVecs(T, NULL, &xi);

	// std::cout << "start EPS solver" << std::endl;

	// EPSCreate(geo->comm, &eps);
	// EPSSetOperators(eps, T, NULL);
	// EPSSetFromOptions(eps);
	// EPSSolve(eps);

	// PetscScalar *eigenValues;
	// for (int i = 0; i < 15; i++){
	// 	EPSGetEigenpair(eps, i, &kr, &ki, xr, xi);
	// 	VecGetArray(xr, &eigenValues);
	// 	for (int j = 0; j < 15; j++){
	// 		if (i == j){
	// 			MatSetValue(lambda, j, i, kr, INSERT_VALUES);
	// 		} else {
	// 			MatSetValue(lambda, j, i, 0, INSERT_VALUES);
	// 		}
	// 		MatSetValue(S, j, i, eigenValues[j], INSERT_VALUES);
	// 	}
	// }

	// MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
	// MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);
	// MatAssemblyBegin(lambda, MAT_FINAL_ASSEMBLY);
	// MatAssemblyEnd(lambda, MAT_FINAL_ASSEMBLY);

	// std::cout << "compute hessian matrix" << std::endl;

	// MatMatMult(Q, S, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &U);
	// MatTranspose(U, MAT_INITIAL_MATRIX, &UTranspose);

	// MatPtAP(lambda, UTranspose, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &ReducedHessian);
	// MatView(ReducedHessian, PETSC_VIEWER_STDOUT_WORLD);

	// std::cout << "convert PETSc to Dolfin" << std::endl;
	// std::shared_ptr<Matrix> out(new Matrix(PETScMatrix(ReducedHessian)));

	// std::cout << "Mat Destroy Reduced Hessian" << std::endl;
	// MatDestroy(&ReducedHessian);

	// std::cout << "hessianMatrix" << std::endl;

	return out;
	
}


// void jetFlowSolver::uncoupledSolFwd(){
// 	std::cout << "uncoupledSolFwd" << std::endl;

// std::shared_ptr<RANSPseudoTimeStepping::Form_J_momentum> J_momentum;
// std::shared_ptr<RANSPseudoTimeStepping::Form_J_rans> J_rans;

// std::shared_ptr<RANSPseudoTimeStepping::CoefficientSpace_momentum> Momentum_STATE;
// std::shared_ptr<RANSPseudoTimeStepping::CoefficientSpace_rans> Rans_STATE;

// std::shared_ptr<RANSPseudoTimeStepping::Form_F_momentum> F_momentum;
// std::shared_ptr<RANSPseudoTimeStepping::Form_F_rans> F_rans;

// 	Momentum_STATE = std::make_shared<RANSPseudoTimeStepping::CoefficientSpace_momentum>(geo->geoMesh);
// 	Rans_STATE = std::make_shared<RANSPseudoTimeStepping::CoefficientSpace_rans>(geo->geoMesh);
// 	F_momentum = std::make_shared<RANSPseudoTimeStepping::Form_F_momentum>(Momentum_STATE);
// 	F_rans = std::make_shared<RANSPseudoTimeStepping::Form_F_rans>(Rans_STATE);
// 	J_momentum = std::make_shared<RANSPseudoTimeStepping::Form_J_momentum>(Momentum_STATE, Momentum_STATE);
// 	J_rans = std::make_shared<RANSPseudoTimeStepping::Form_J_rans>(Rans_STATE, Rans_STATE);

// 	momentum_inflow_boundary = std::make_shared<DirichletBC>(Momentum_STATE->sub(0), u_inflow, geo->boundaryParts, 1);
// 	momentum_axis_boundary   = std::make_shared<DirichletBC>(Momentum_STATE->sub(0)->sub(1), zero_scaler, geo->boundaryParts, 2);
// 	rans_k_inflow_boundary = std::make_shared<DirichletBC>(Rans_STATE->sub(0), k_inflow, geo->boundaryParts, 1);
// 	rans_e_inflow_boundary = std::make_shared<DirichletBC>(Rans_STATE->sub(1), e_inflow, geo->boundaryParts, 1);

// 	bcs_momentum  = {momentum_inflow_boundary.get(), momentum_axis_boundary.get()};			
// 	bcs0_momentum = {momentum_homogenize.get(), momentum_axis_homogenize.get()};
// 	bcs_rans  = {rans_k_inflow_boundary.get(), rans_e_inflow_boundary.get()};			
// 	bcs0_rans = {k_homogenize.get(), e_homogenize.get()};

// 	momentum = std::make_shared<Function>(Momentum_STATE);
// 	rans     = std::make_shared<Function>(Rans_STATE);

// 	if (fileExists(data_file)){
// 		std::cout << "reading input file" << std::endl;
// 		auto inputMesh = std::make_shared<Mesh>();
// 		HDF5File inputFile(MPI_COMM_WORLD, data_file, "r");
// 		inputFile.read(*inputMesh.get(), "/mesh", false);
// 		auto inputState = std::make_shared<RANSPseudoTimeStepping::CoefficientSpace_x>(inputMesh);
// 		auto inputData = std::make_shared<Function>(inputState);
// 		inputFile.read(*inputData.get(), "/states");
// 		LagrangeInterpolator::interpolate(*x.get(), *inputData.get());
// 		assign(xl, x);
// 		inputFile.close();
// 	} else{
// 		initCond zeroInit;
// 		x->interpolate(zeroInit);
// 		assign(xl, x);
// 	}

// 	assign({u, p, k, e}, x);
// 	assign(momentum, {u, p});
// 	assign(rans, {k, e});
// 	assign(xl, {u, p, k, e});

// 	m->interpolate(mExpression);
// 	double sigmaUpdate = 20.0;
// 	sigma.reset(new Constant(sigmaUpdate)); 

// 	Parameters params("nonlinear_variational_solver");
// 	Parameters newton_params("newton_solver");
// 	newton_params.add("relative_tolerance", 1e-3);
// 	newton_params.add("convergence_criterion", "residual");
// 	newton_params.add("error_on_nonconvergence", false);
// 	newton_params.add("maximum_iterations", 20);
// 	params.add(newton_params);

// 	for (int i = 0; i < 10000; ++i){
// 		sigma.reset(new Constant(sigmaUpdate)); 
// 		F_momentum->sigma = sigma;
// 		F_momentum->m     = m;
// 		F_momentum->xl    = xl;
// 		F_momentum->ds    = geo->boundaryParts;
// 		F_momentum->momentum = momentum;  

// 		J_momentum->sigma = sigma;
// 		J_momentum->m     = m;
// 		J_momentum->xl    = xl;
// 		J_momentum->ds    = geo->boundaryParts;
// 		J_momentum->momentum = momentum;

// 		solve(*F_momentum == 0, *momentum, bcs_momentum, *J_momentum, params);

// 		assign({u, p}, momentum);
// 		assign(xl, {u, p, k, e});

// 		assign(x, xl);
// 		updateInitialState();

// 		for (int n = 0; n < 5; ++n){
// 			F_rans->sigma = sigma;
// 			F_rans->m     = m;
// 			F_rans->xl    = xl;
// 			F_rans->ds    = geo->boundaryParts;
// 			F_rans->rans  = rans;  

// 			J_rans->sigma = sigma;
// 			J_rans->m     = m;
// 			J_rans->xl    = xl;
// 			J_rans->ds    = geo->boundaryParts;
// 			J_rans->rans  = rans;

// 			solve(*F_rans == 0, *rans, bcs_rans, *J_rans, params);

// 			assign({k, e}, rans);
// 			assign(xl, {u, p, k, e});

// 			assign(x, xl);
// 			updateInitialState();
// 		}
// 	}

// };


std::shared_ptr<Matrix> jetFlowSolver::dolfinTranspose(const GenericMatrix & A){
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

std::shared_ptr<Matrix> jetFlowSolver::dolfinMatMatMult(const GenericMatrix & A, const GenericMatrix & B)
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

std::shared_ptr<Matrix> jetFlowSolver::dolfinMatTransposeMatMult(const GenericMatrix & A, const GenericMatrix & B)
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

void dolfinVecView(MPI_Comm commObj, Vec vecObj){
	PetscViewer viewer;
	PetscViewerBinaryOpen(commObj, "vecObj", FILE_MODE_WRITE, &viewer);
	VecView(vecObj,viewer);
	PetscViewerDestroy(&viewer);	
}

void dolfinMatView(MPI_Comm commObj, Mat matObj){
	PetscViewer viewer;
	PetscViewerBinaryOpen(commObj, "matObj", FILE_MODE_WRITE, &viewer);
	MatView(matObj,viewer);
	PetscViewerDestroy(&viewer);
}