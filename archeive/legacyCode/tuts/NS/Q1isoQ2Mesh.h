class mesh_periodic{
private:
public:
	int level;
	int division_Q1isoQ2;
	int division_Q1;
	int num_Q1isoQ2_element;
	int num_Q1_element;
	int num_node_Q1isoQ2;
	int num_node_Q1;

	std::unique_ptr<std::unique_ptr<double[]>[]> points;
	std::unique_ptr<std::unique_ptr<int[]>[]> quadrilaterals;
	std::unique_ptr<int[]> Q1_idx;
	std::unique_ptr<int[]> mesh_idx;
	std::unique_ptr<int[]> PBC_Q1isoQ2_idx;
	std::unique_ptr<int[]> PBC_Q1_idx;
	std::unique_ptr<int[]> bs;

	Mat A;
	Mat M;
	Mat zeroMatrix;

	mesh_periodic(int level_);
	~mesh_periodic(){};

	void mass_matrix_element(double P[][4], double mass_element[][4]);
	void stiffness_matrix_element(double P[][4], double stiffness_element[][4]);
	void M_matrix();
	void A_matrix();
	void load_vector_element(double x1[], double x2[], double v[], int dir, double (*f)(double, double, int, void*), void* ctx);
	void load_vector(Vec &load, double (*f)(double, double, int, void*), void* ctx);
	void interpolate(Vec &load, Vec &interpolation);

	void shape_function(double epsilon, double eta, double N[]);
	void jacobian_matrix(double x1[4], double x2[4], double epsilon, double eta, double J[2][2]);
	void jacobian_inv(double x1[4], double x2[4], double epsilon, double eta, double J[2][2]);
	void basis_function(double epsilon, double eta, double N[]);
	void basis_function_derivative(double dPhi[4][2], double epsilon, double eta);
	void hat_function_derivative(double dPhi[4][2], double epsilon, double eta, double x1[4], double x2[4]);
	double shape_interpolation(double epsilon, double eta, double x[4]);
	double jacobian_det(double J[2][2]);
};

mesh_periodic::mesh_periodic(int level_): level(level_) {
	division_Q1isoQ2 = 2*(std::pow(2, level_+1));
	division_Q1 = division_Q1isoQ2/2;

	num_Q1isoQ2_element = division_Q1isoQ2*division_Q1isoQ2;
	num_Q1_element = 0.25*num_Q1isoQ2_element;
	num_node_Q1isoQ2 = std::pow(division_Q1isoQ2+1, 2);
	num_node_Q1 = std::pow(division_Q1isoQ2/2+1, 2);

	points    = std::make_unique<std::unique_ptr<double[]>[]>(2);
	points[0] = std::make_unique<double[]>(num_node_Q1isoQ2);
	points[1] = std::make_unique<double[]>(num_node_Q1isoQ2);

	double xCoord = 0;
	double yCoord = 0;
	for (int i=0; i<num_node_Q1isoQ2; i++){
		if (xCoord-1 > 1e-6){
			xCoord = 0;
			yCoord += 1.0/division_Q1isoQ2;
		}
		points[0][i] = xCoord;
		points[1][i] = yCoord;
		xCoord += 1.0/division_Q1isoQ2;
	}

	quadrilaterals = std::make_unique<std::unique_ptr<int[]>[]>(4);
	for (int i=0; i<4; i++){
		quadrilaterals[i] = std::make_unique<int[]>(num_Q1isoQ2_element);
	}
	int refDof[4] = {0, 1, division_Q1isoQ2+2, division_Q1isoQ2+1};
	for (int i=0; i<num_Q1isoQ2_element; i++){
		quadrilaterals[0][i] = refDof[0];
		quadrilaterals[1][i] = refDof[1];
		quadrilaterals[2][i] = refDof[2];
		quadrilaterals[3][i] = refDof[3];

		if ((refDof[1]+1)%(division_Q1isoQ2+1) == 0){
			refDof[0] += 2;
			refDof[1] += 2;
			refDof[2] += 2;
			refDof[3] += 2;
		} else {
			refDof[0] += 1;
			refDof[1] += 1;
			refDof[2] += 1;
			refDof[3] += 1;
		}
	}

	Q1_idx = std::make_unique<int[]>(num_node_Q1isoQ2);
	for (int i = 0; i < num_node_Q1isoQ2; i++){
		Q1_idx[i] = num_node_Q1isoQ2;
	}

	int position = 0;
	int value = 0;
	for (int i = 0; i < division_Q1isoQ2/2.0+1; i++){
		position = 2*(division_Q1isoQ2+1)*i;
		for (int j = 0; j < division_Q1isoQ2/2.0+1; j++){
			Q1_idx[position] = value;
			value += 1; 
			position = position + 2;
		}
	}

	for (int i = 0; i < num_node_Q1isoQ2; i++){
		if (Q1_idx[i] == num_node_Q1isoQ2){
			Q1_idx[i] += i;
		}
	}

	std::vector<int> mesh_idx_vector(num_node_Q1isoQ2);
	std::vector<int> mesh_idx_vector2(num_node_Q1isoQ2);

	std::iota(mesh_idx_vector.begin(), mesh_idx_vector.end(), 0);
	std::iota(mesh_idx_vector2.begin(), mesh_idx_vector2.end(), 0);

	std::stable_sort(mesh_idx_vector.begin(), mesh_idx_vector.end(), [&](int i, int j){return Q1_idx[i] < Q1_idx[j];});
	std::stable_sort(mesh_idx_vector2.begin(), mesh_idx_vector2.end(), [&](int i, int j){return mesh_idx_vector[i] < mesh_idx_vector[j];});
	mesh_idx = std::make_unique<int[]>(num_node_Q1isoQ2);
	for (int i = 0; i < num_node_Q1isoQ2; i++){
		mesh_idx[i] = mesh_idx_vector2[i];
	}

	//Periodic Boundary Condition Mapping
	std::vector<int> PBC_Q1isoQ2(num_node_Q1isoQ2);
	std::vector<int> PBC_Q1(num_node_Q1isoQ2);

	std::iota(PBC_Q1isoQ2.begin(), PBC_Q1isoQ2.end(), 0);
	for (int i = 0; i < num_node_Q1isoQ2; ++i){
		PBC_Q1[i] = mesh_idx[i];
	}

	bs = std::make_unique<int[]>(division_Q1isoQ2*2+1);
	int b_pos[2] = {division_Q1isoQ2, num_node_Q1isoQ2-1};
	bs[0] = b_pos[0];
	bs[1] = b_pos[1];
	bs[division_Q1isoQ2*2] = num_node_Q1isoQ2-1-division_Q1isoQ2;	 
	PBC_Q1isoQ2[b_pos[0]] = 0;
	PBC_Q1isoQ2[b_pos[1]] = 0;
	PBC_Q1isoQ2[num_node_Q1isoQ2-1-division_Q1isoQ2] = 0;
	PBC_Q1[b_pos[0]] = mesh_idx[0];
	PBC_Q1[b_pos[1]] = mesh_idx[0];
	PBC_Q1[num_node_Q1isoQ2-1-division_Q1isoQ2] = mesh_idx[0];
	for (int i = 1; i < division_Q1isoQ2;++i){
		PBC_Q1isoQ2[b_pos[0]+(division_Q1isoQ2+1)*i] = b_pos[0]+(division_Q1isoQ2+1)*i - division_Q1isoQ2;
		PBC_Q1isoQ2[b_pos[1]-i] = division_Q1isoQ2 - i;
		PBC_Q1[b_pos[0]+(division_Q1isoQ2+1)*i] = mesh_idx[b_pos[0]+(division_Q1isoQ2+1)*i - division_Q1isoQ2];
		PBC_Q1[b_pos[1]-i] = mesh_idx[division_Q1isoQ2 - i];
		bs[2*i] = b_pos[0]+(division_Q1isoQ2+1)*i;
		bs[2*i+1] = b_pos[1]-i;
	}

	PBC_Q1isoQ2_idx = std::make_unique<int[]>(num_node_Q1isoQ2);
	PBC_Q1_idx = std::make_unique<int[]>(num_node_Q1isoQ2);
	for (int i = 0; i < num_node_Q1isoQ2; i++){
		PBC_Q1isoQ2_idx[i] = PBC_Q1isoQ2[i];
		PBC_Q1_idx[i] = PBC_Q1[i];
	}	
}

void mesh_periodic::shape_function(double epsilon, double eta, double N[]){
	N[0] = 0.25*(1.0-epsilon)*(1.0-eta);
	N[1] = 0.25*(1.0+epsilon)*(1.0-eta);
	N[2] = 0.25*(1.0+epsilon)*(1.0+eta);
	N[3] = 0.25*(1.0-epsilon)*(1.0+eta);
};

double mesh_periodic::shape_interpolation(double epsilon, double eta, double x[4]){
	double N[4];
	double x_interpolated;
	shape_function(epsilon, eta, N);
	x_interpolated = N[0]*x[0]+N[1]*x[1]+N[2]*x[2]+N[3]*x[3];
	return x_interpolated;
}

void mesh_periodic::jacobian_matrix(double x1[4], double x2[4], double eta, double epsilon, double J[2][2]){
	J[0][0] = 0.25*((eta-1)*x1[0]+(1-eta)*x1[1]+(1+eta)*x1[2]-(1+eta)*x1[3]); 
	J[0][1] = 0.25*((eta-1)*x2[0]+(1-eta)*x2[1]+(1+eta)*x2[2]-(1+eta)*x2[3]); 
	J[1][0] = 0.25*((epsilon-1)*x1[0]-(1+epsilon)*x1[1]+(1+epsilon)*x1[2]+(1-epsilon)*x1[3]); 
	J[1][1] = 0.25*((epsilon-1)*x2[0]-(1+epsilon)*x2[1]+(1+epsilon)*x2[2]+(1-epsilon)*x2[3]); 
};

double mesh_periodic::jacobian_det(double J[2][2]){
	double detJ = J[1][1]*J[0][0] - J[0][1]*J[1][0];
	return detJ;
};

void mesh_periodic::jacobian_inv(double x1[4], double x2[4], double epsilon, double eta, double Jinv[2][2]){
	double J[2][2];
	jacobian_matrix(x1, x2, epsilon, eta, J);
	double Jdet = jacobian_det(J);
	Jinv[0][0] = J[1][1]/Jdet;	
	Jinv[0][1] =-J[0][1]/Jdet;
	Jinv[1][0] =-J[1][0]/Jdet;	
	Jinv[1][1] = J[0][0]/Jdet;
};

void mesh_periodic::basis_function(double epsilon, double eta, double N[]){
	shape_function(epsilon, eta, N);
};

void mesh_periodic::basis_function_derivative(double basisdPhi[4][2], double epsilon, double eta){
	basisdPhi[0][0] = -0.25+0.25*eta;	
	basisdPhi[1][0] = 0.25-0.25*eta;
	basisdPhi[2][0] = 0.25+0.25*eta;
	basisdPhi[3][0] = -0.25-0.25*eta;
	basisdPhi[0][1] = -0.25+0.25*epsilon;	
	basisdPhi[1][1] = -0.25-0.25*epsilon;
	basisdPhi[2][1] = 0.25+0.25*epsilon;
	basisdPhi[3][1] = 0.25-0.25*epsilon;		
};

void mesh_periodic::hat_function_derivative(double dPhi[4][2], double epsilon, double eta, double x1[4], double x2[4]){
	double basisdPhi[4][2];
	double Jinv[2][2];
	basis_function_derivative(basisdPhi, epsilon, eta);
	jacobian_inv(x1, x2, epsilon, eta, Jinv);
	for (int i = 0; i < 4; ++i){
		for (int j = 0; j < 2; ++j){
			dPhi[i][j] = basisdPhi[i][0]*Jinv[0][j] + basisdPhi[i][1]*Jinv[1][j];
		}
	}
}

void mesh_periodic::mass_matrix_element(double P[][4], double mass_element[][4]){
	double N[4];
	double J[2][2];
	double Jdet[4];

	double refPoints[2];
	double refWeights[2];

	refPoints[0] = -1./sqrt(3.0);
	refPoints[1] =  1./sqrt(3.0);

	refWeights[0] = 1.0;
	refWeights[1] = 1.0;

	double gpPoints[4][4];
	double gpWeights[4];

	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 2; j++){
			basis_function(refPoints[i], refPoints[j], N);
			jacobian_matrix(P[0], P[1], refPoints[i], refPoints[j], J);
			gpWeights[2*i+j] = refWeights[i]*refWeights[j];
			Jdet[2*i+j] = jacobian_det(J);
			for (int k = 0; k < 4; ++k){
				gpPoints[2*i+j][k] = N[k];
			}
		}
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			mass_element[i][j] = 0;
			for (int k = 0; k < 4; k++){
				mass_element[i][j] += gpWeights[k]*gpPoints[k][i]*gpPoints[k][j]*Jdet[k];
			}
		}
	}
}

void mesh_periodic::M_matrix(){
	MatCreate(PETSC_COMM_SELF, &M);
	MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, num_node_Q1isoQ2, num_node_Q1isoQ2);
	if (level > 7){
		MatSetType(M, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(M, MATSEQAIJ);
	}
	MatSeqAIJSetPreallocation(M, 12, NULL);
	MatMPIAIJSetPreallocation(M, 12, NULL, 12, NULL);

	double mass_element[4][4];
	double element_points[2][4];
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		mass_matrix_element(element_points, mass_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
 				MatSetValue(M, PBC_Q1isoQ2_idx[quadrilaterals[m][i]], PBC_Q1isoQ2_idx[quadrilaterals[n][i]], mass_element[m][n], ADD_VALUES);
			}
		}			
	}
	MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

void mesh_periodic::stiffness_matrix_element(double P[][4], double stiffness_element[][4]){
	double dPhi[4][2];
	double J[2][2];

	double refPoints[2];
	double refWeights[2];

	refPoints[0] = -1./sqrt(3.0);
	refPoints[1] =  1./sqrt(3.0);

	refWeights[0] = 1.0;
	refWeights[1] = 1.0;

	double gpPoints[4][8];
	double gpWeights[4];
	double Jdet[4];

	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 2; j++){
			hat_function_derivative(dPhi, refPoints[i], refPoints[j], P[0], P[1]);
			jacobian_matrix(P[0], P[1], refPoints[i], refPoints[j], J);
			gpWeights[2*i+j] = refWeights[i]*refWeights[j];
			Jdet[2*i+j] = jacobian_det(J);
			for (int k = 0; k < 8; ++k){
				gpPoints[2*i+j][k] = dPhi[k%4][k/4];
			}
		}
	}

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			stiffness_element[i][j] = 0;
			for (int k = 0; k < 4; k++){
				stiffness_element[i][j] += gpWeights[k]*(gpPoints[k][i]*gpPoints[k][j]+gpPoints[k][4+i]*gpPoints[k][4+j])*Jdet[k];
			}
		}
	}
};

void mesh_periodic::A_matrix(){
	MatCreate(PETSC_COMM_SELF, &A);
	MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, num_node_Q1isoQ2, num_node_Q1isoQ2);
	if (level > 7){
		MatSetType(A, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(A, MATSEQAIJ);
	}
	MatSeqAIJSetPreallocation(A, 12, NULL);
	MatMPIAIJSetPreallocation(A, 12, NULL, 12, NULL);


	double stiffness_element[4][4];
	double element_points[2][4];
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		stiffness_matrix_element(element_points, stiffness_element);
		for (int m = 0; m < 4; m++){
			for (int n = 0; n < 4; n++){
 				MatSetValue(A, PBC_Q1isoQ2_idx[quadrilaterals[m][i]], PBC_Q1isoQ2_idx[quadrilaterals[n][i]], stiffness_element[m][n], ADD_VALUES);
			}
		}			
	}
	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
};

void mesh_periodic::load_vector_element(double x1[], double x2[], double v[], int dir, double (*f)(double, double, int, void*), void* ctx){
	double J[2][2];

	double refPoints[5];
	double refWeights[5];

	refPoints[0] = -1./3.*sqrt(5.+2.*sqrt(10./7.));
	refPoints[1] = -1./3.*sqrt(5.-2.*sqrt(10./7.));
	refPoints[2] = 0.;
	refPoints[3] = 1./3.*sqrt(5.-2.*sqrt(10./7.));
	refPoints[4] = 1./3.*sqrt(5.+2.*sqrt(10./7.));

	refWeights[0] = (322.-13.*sqrt(70.))/900.;
	refWeights[1] = (322.+13.*sqrt(70.))/900.;
	refWeights[2] = 128./225.;
	refWeights[3] = (322.+13.*sqrt(70.))/900.;
	refWeights[4] = (322.-13.*sqrt(70.))/900.;

	double gpPoints1[25];
	double gpPoints2[25][4];
	double gpWeights[25];
	double Jdet[25];
	double N[4];

	double x;
	double y;

	for (int i = 0; i < 5; i++){
		for (int j = 0; j < 5; j++){
			basis_function(refPoints[i], refPoints[j], N);
			jacobian_matrix(x1, x2, refPoints[i], refPoints[j], J);
			x = shape_interpolation(refPoints[i], refPoints[j], x1);
			y = shape_interpolation(refPoints[i], refPoints[j], x2);
			gpWeights[5*i+j] = refWeights[i]*refWeights[j];
			Jdet[5*i+j] = jacobian_det(J);
			gpPoints1[5*i+j] = f(x, y, dir, ctx);
			for (int k = 0; k < 4; ++k){
				gpPoints2[5*i+j][k] = N[k];
			}
		}
	}
	for (int i = 0; i < 4; i++){
		v[i] = 0;
		for (int k = 0; k < 25; k++){
			v[i] += gpWeights[k]*gpPoints1[k]*gpPoints2[k][i]*Jdet[k];
		}
	}
};

void mesh_periodic::load_vector(Vec &load, double (*f)(double, double, int, void*), void* ctx){
 	VecCreate(PETSC_COMM_SELF, &load);
	VecSetSizes(load, PETSC_DECIDE, 2*num_node_Q1isoQ2+num_node_Q1);
	if (level > 7){
		VecSetType(load, VECSEQCUDA);
	} else {
		VecSetType(load, VECSEQ);
	}

	double v_elementx[4];
	double v_elementy[4];
	double element_points[2][4];
	for (int i = 0; i < num_Q1isoQ2_element; i++){
		for (int j = 0; j < 4; j++){
			element_points[0][j] = points[0][quadrilaterals[j][i]];
			element_points[1][j] = points[1][quadrilaterals[j][i]];
		}
		load_vector_element(element_points[0], element_points[1], v_elementx, 0, f, ctx);
		load_vector_element(element_points[0], element_points[1], v_elementy, 1, f, ctx);
		for (int k = 0; k < 4; k++){
			VecSetValue(load, PBC_Q1isoQ2_idx[quadrilaterals[k][i]], v_elementx[k], ADD_VALUES);
			VecSetValue(load, num_node_Q1isoQ2+PBC_Q1isoQ2_idx[quadrilaterals[k][i]], v_elementy[k], ADD_VALUES);
		}
	}

	VecAssemblyBegin(load);
	VecAssemblyEnd(load);
};

void mesh_periodic::interpolate(Vec &load, Vec &interpolation){
	MatCreate(PETSC_COMM_SELF, &zeroMatrix);
	MatSetSizes(zeroMatrix, PETSC_DECIDE, PETSC_DECIDE, num_node_Q1, num_node_Q1);
	if (level > 7){
		MatSetType(zeroMatrix, MATSEQAIJCUSPARSE);
	} else {
		MatSetType(zeroMatrix, MATSEQAIJ);
	}
	MatSeqAIJSetPreallocation(zeroMatrix, 12, NULL);
	MatMPIAIJSetPreallocation(zeroMatrix, 12, NULL, 12, NULL);
	for (int i = 0; i < num_node_Q1; ++i){
		MatSetValue(zeroMatrix, i, i, 0, INSERT_VALUES);
	}
	MatAssemblyBegin(zeroMatrix, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(zeroMatrix, MAT_FINAL_ASSEMBLY);

	Mat workspaceM;
	Mat massfunction[] = {M, NULL, NULL, NULL, M, NULL, NULL, NULL, zeroMatrix};
	MatCreateNest(PETSC_COMM_SELF, 3, NULL, 3, NULL, massfunction, &workspaceM);	

	KSP InterpolationOperator;
	PC  InterpolationPC;
	KSPCreate(PETSC_COMM_WORLD, &InterpolationOperator);
	KSPSetType(InterpolationOperator, KSPGMRES);
	KSPSetOperators(InterpolationOperator, workspaceM, workspaceM);
	KSPSetTolerances(InterpolationOperator, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPGetPC(InterpolationOperator, &InterpolationPC);
	PCSetType(InterpolationPC, PCBJACOBI);

	KSPSolve(InterpolationOperator, load, interpolation);
	MatDestroy(&workspaceM);
	KSPDestroy(&InterpolationOperator);
};

