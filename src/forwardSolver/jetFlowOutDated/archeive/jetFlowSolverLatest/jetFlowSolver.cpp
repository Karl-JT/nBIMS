#include "jetFlowSolver.h"

jetFlowGeo::jetFlowGeo(int nx_, int ny_): nx(nx_), ny(ny_), P0(2, point0), P1(2, point1){
	geoMesh = std::make_shared<RectangleMesh>(P0, P1, nx, ny);
	

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
};

void jetFlowSolver::setFEM(){
	auto Vh_STATE = std::make_shared<RANS::FunctionSpace>(geo->geoMesh);

	auto zero_scaler = std::make_shared<Constant>(0.0);
	auto zero_vector = std::make_shared<Constant>(0.0, 0.0);
	auto u_inflow    = std::make_shared<uBoundary>();
	auto k_inflow    = std::make_shared<kBoundary>();
	auto e_inflow    = std::make_shared<eBoundary>();

	auto u_inflow_boundary = std::make_shared<DirichletBC>(Vh_STATE->sub(0), u_inflow, geo->boundaryParts, 1);
	auto k_inflow_boundary = std::make_shared<DirichletBC>(Vh_STATE->sub(2), k_inflow, geo->boundaryParts, 1);
	auto e_inflow_boundary = std::make_shared<DirichletBC>(Vh_STATE->sub(3), e_inflow, geo->boundaryParts, 1);	
	auto axis_boundary     = std::make_shared<DirichletBC>(Vh_STATE->sub(0)->sub(1), zero_scaler, geo->boundaryParts, 2);

	auto u_homogenize    = std::make_shared<DirichletBC>(Vh_STATE->sub(0), zero_vector, geo->boundaryParts, 1);
	auto k_homogenize    = std::make_shared<DirichletBC>(Vh_STATE->sub(2), zero_scaler, geo->boundaryParts, 1);
	auto e_homogenize    = std::make_shared<DirichletBC>(Vh_STATE->sub(3), zero_scaler, geo->boundaryParts, 1);	
	auto axis_homogenize = std::make_shared<DirichletBC>(Vh_STATE->sub(0)->sub(1), zero_scaler, geo->boundaryParts, 2);

	std::vector<const DirichletBC*> bcs  = {u_inflow_boundary.get(), k_inflow_boundary.get(), e_inflow_boundary.get(), axis_boundary.get()};			
	std::vector<const DirichletBC*> bcs0 = {u_homogenize.get(), k_homogenize.get(), e_homogenize.get(), axis_homogenize.get()};

	auto x     = std::make_shared<Function>(Vh_STATE);
	auto xl    = std::make_shared<Function>(Vh_STATE);
	auto m     = std::make_shared<Function>(Vh_STATE->sub(2)->collapse());
	auto u_ff  = std::make_shared<Constant>(0.0, 0.0);
	
	double timeStep = 0.1;
	auto sigma = std::make_shared<Constant>(1.0/timeStep);

	auto u = std::make_shared<Function>(Vh_STATE->sub(0)->collapse());
	auto p = std::make_shared<Function>(Vh_STATE->sub(1)->collapse());
	auto k = std::make_shared<Function>(Vh_STATE->sub(2)->collapse());
	auto e = std::make_shared<Function>(Vh_STATE->sub(3)->collapse());

	std::string data_name = "states";
	std::string data_file = "states.h5";
	// if (fileExists(data_file)){
	// 	HDF5File inputFile(geo->geoMesh->mpi_comm(), "states.h5", "r");
	// 	inputFile.read(*x.get(), data_name);

	// 	File uFile("u.pvd");
	// 	File kFile("k.pvd");
	// 	assign({u, p, k, e}, x);
	// 	uFile << *u;
	// 	kFile << *k;
	// 	assign(xl, x);
	// 	inputFile.close();
	// } else{
	// 	initCond zeroInit;
	// 	x->interpolate(zeroInit);
	// 	xl->interpolate(zeroInit);
	// }
	initCond zeroInit;
	x->interpolate(zeroInit);
	xl->interpolate(zeroInit);

	cout << (*x)[2].str(1) << endl;

	initM zeroM;
	m->interpolate(zeroM);

	Vector r;

	RANS::ResidualForm F(Vh_STATE);
	F.sigma = sigma;
	F.u_ff = u_ff;
	F.m  = m;
	F.x  = x;  
	F.xl = xl;
	F.ds = geo->boundaryParts;

	// cout << F.rank() << endl;
	// cout << F.num_coefficients() << endl;
	auto funspace = F.function_spaces();
	auto cff = F.coefficients();
	auto ufcform = F.ufc_form();

	std::vector<double> forPrint;


	// cff[0]->compute_vertex_values(forPrint, *(geo->geoMesh));
	// for (int i = 0; i < forPrint.size(); i++){
	// 	cout << forPrint[i] << " ";
	// }
	// cout << endl << endl;
	// cff[1]->compute_vertex_values(forPrint, *(geo->geoMesh));
	// for (int i = 0; i < forPrint.size(); i++){
	// 	cout << forPrint[i] << " ";
	// }
	cout << endl << endl;
	cff[2]->compute_vertex_values(forPrint, *(geo->geoMesh));
	for (int i = 0; i < forPrint.size(); i++){
		cout << forPrint[i] << " ";
	}
	cout << endl << endl;
	// cff[3]->compute_vertex_values(forPrint, *(geo->geoMesh));
	// for (int i = 0; i < forPrint.size(); i++){
	// 	cout << forPrint[i] << " ";
	// }
	// cout << endl << endl;
	// cff[4]->compute_vertex_values(forPrint, *(geo->geoMesh));
	// for (int i = 0; i < forPrint.size(); i++){
	// 	cout << forPrint[i] << " ";
	// }
	// cout << endl << endl;

	//cout << F.ds->str(1);
	std::vector<std::size_t> test;

	RANS::JacobianForm J(Vh_STATE, Vh_STATE);
	J.sigma = sigma;
	J.u_ff = u_ff;
	J.m = m;
	J.x = x;
	J.xl = xl;
	J.ds = geo->boundaryParts;

	Parameters params("nonlinear_variational_solver");
	Parameters newton_params("newton_solver");
	newton_params.add("relative_tolerance", 1e-10);
	newton_params.add("convergence_criterion", "residual");
	newton_params.add("error_on_nonconvergence", false);
	newton_params.add("maximum_iterations", 5);
	params.add(newton_params);

	double r_norm;

	assemble(r, F);
	for (std::size_t bcCount = 0; bcCount < bcs.size(); bcCount++){
		bcs[bcCount]->apply(r);
	}
	// for (int i = 0; i < r.size(); i++){
	// 	if (i%4 == 0){
	// 		cout << endl;
	// 	}
	// 	cout << r[i] << " ";
	// }
	r_norm = norm(r, "l2");
	cout << "norm: " << r_norm << endl; // << " norm/sigma: " << r_norm*timeStep << endl;

	solve(F == 0, *x, bcs, J, params);

	// for (int i = 0; i < 10000; i++){
	// 	// if (r_norm*timeStep < 0.0001){
	// 	// 	timeStep = timeStep*1.0;
	// 	// 	sigma.reset(new Constant(1.0/timeStep));			
	// 	// }
	// 	// sigma.reset(new Constant(0.0));
	// 	assign(xl, x);

	// 	RANS::ResidualForm F(Vh_STATE);
	// 	F.sigma = sigma;
	// 	F.u_ff  = u_ff;
	// 	F.m  = m;
	// 	F.x  = x;
	// 	F.xl = xl;
	// 	F.ds = geo->boundaryParts;

	// 	RANS::JacobianForm J(Vh_STATE, Vh_STATE);
	// 	J.sigma = sigma;
	// 	J.u_ff = u_ff;
	// 	J.m  = m;
	// 	J.x  = x; 	
	// 	J.xl = xl;
	// 	J.ds = geo->boundaryParts;

	// 	assemble(r, F);
	// 	for (std::size_t bcCount = 0; bcCount < bcs.size(); bcCount++){
	// 		bcs[bcCount]->apply(r);
	// 	}
	// 	r_norm = norm(r, "l2");
	// 	cout << "norm: " << r_norm << endl; // << " norm/sigma: " << r_norm*timeStep << endl;

	// 	// sigma.reset(new Constant(1.0/timeStep));
	// 	// F.sigma = sigma;
	// 	// J.sigma = sigma;
	//  	solve(F == 0, *x, bcs, J, params);
	// }
	// HDF5File outputFile(geo->geoMesh->mpi_comm(), "states.h5", "w");
	// outputFile.write(*x.get(), "states");
	// outputFile.close();
};

void jetFlowSolver::setPDE(){

};

void jetFlowSolver::soluFwd(){
		
};