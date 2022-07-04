#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

#include <confIO.h>
#include <elasticMixed2d.h>
// #include <MLMCMC_Bi_Uniform_Dummy.h>
#include <MLMCMC_Bi_Dummy.h>


void read_config(std::vector<std::string> &paras, std::vector<std::string> &vals){
	std::ifstream cFile("mlmcmc_config.txt");
	if (cFile.is_open()){
		std::string line;
		while(getline(cFile, line)){
			line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
			if (line[0] == '#' || line.empty()){
				continue;
			}
			auto delimiterPos = line.find("=");
			auto name = line.substr(0, delimiterPos);
			auto value = line.substr(delimiterPos+1);

			paras.push_back(name);
			vals.push_back(value);
		}
	}
	cFile.close();
}

int main(int argc, char* argv[])
{
	int rank;
	int size;

	// // Solver Options List 1
	// PetscOptionsSetValue(NULL, "-ksp_type", "fgmres");
	// PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_type", "gamg");


	// // Solver Options List 1
	// PetscOptionsSetValue(NULL, "-ksp_type", "minres");
	// PetscOptionsSetValue(NULL, "-pc_type", "jacobi");

	// // Solver Options List 2
	PetscOptionsSetValue(NULL, "-pc_fieldsplit_detect_saddle_point", "1");
	PetscOptionsSetValue(NULL, "-ksp_type", "fgmres");
	PetscOptionsSetValue(NULL, "-pc_fieldsplit_type", "schur");
	PetscOptionsSetValue(NULL, "-pc_fieldsplit_schur_fact_type", "full");
	PetscOptionsSetValue(NULL, "-fieldsplit_0_ksp_type", "cg");
	//PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_type", "hypre");
	PetscOptionsSetValue(NULL, "-fieldsplit_1_ksp_type", "cg");
	PetscOptionsSetValue(NULL, "-pc_fieldsplit_schur_precondition", "selfp");
	PetscOptionsSetValue(NULL, "-fieldsplit_1_pc_type", "ilu");

	// Solver Options List 3
	//PetscOptionsSetValue(NULL, "-ksp_type", "gmres");
	//PetscOptionsSetValue(NULL, "-pc_fieldsplit_type", "schur");
	//PetscOptionsSetValue(NULL, "-fieldsplit_0_ksp_type", "preonly");
	//PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_type", "ilu");
	//PetscOptionsSetValue(NULL, "-fieldsplit_1_ksp_type", "preonly");
        //PetscOptionsSetValue(NULL, "-pc_fieldsplit_schur_precondition", "selfp");
	//PetscOptionsSetValue(NULL, "-fieldsplit_1_pc_type", "jacobi");

	//// Solver Options List 4 direct solver mumps
	//PetscOptionsSetValue(NULL, "-ksp_type", "preonly");
	//PetscOptionsSetValue(NULL, "-ksp_error_if_not_converged", "1");
	//PetscOptionsSetValue(NULL, "-pc_type", "cholesky");
	//PetscOptionsSetValue(NULL, "-pc_factor_mat_solver_type", "mumps");
	////PetscOptionsSetValue(NULL, "-mat_mumps_icntl_1", "1");
	////PetscOptionsSetValue(NULL, "-mat_mumps_icntl_2", "1");
	////PetscOptionsSetValue(NULL, "-mat_mumps_icntl_3", "1");
	////PetscOptionsSetValue(NULL, "-mat_mumps_icntl_4", "3");
	//PetscOptionsSetValue(NULL, "-mat_mumps_icntl_28", "1");
	//PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "2");
	//PetscOptionsSetValue(NULL, "-mat_mumps_icntl_24", "1");
	//PetscOptionsSetValue(NULL, "-mat_mumps_cntl_3", "1e-16");

	PetscInitialize(&argc, &argv, NULL, NULL);
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

    	conf         confData;
    	read_config(&confData);

	if (rank == 0){
        display_config(&confData);
	}

	switch (confData.task){
        case 0: //Generate Observation
        {
            elasticMixed2dSolver singleForwardSolver(MPI_COMM_SELF, confData.levels, confData.num_term,confData.noiseVariance);
	    std::cout << "confData.num_term: " << confData.num_term << std::endl;
	    for (int i = 0; i < confData.num_term; i++){
	        singleForwardSolver.samples[i]=confData.rand_coef[i];
		std::cout << "coefficient" << i << ": " << singleForwardSolver.samples[i] << std::endl;
	    }
            singleForwardSolver.solve();
            std::cout << singleForwardSolver.ObsOutput() << std::endl;
            std::cout << singleForwardSolver.QoiOutput() << std::endl;
            break;
        }

        case 1:  //plain MCMC
	{
            std::default_random_engine generator;
	    generator.seed(confData.randomSeed);
	    double qoimean;
	    elasticMixed2dSolver singleForwardSolver(MPI_COMM_SELF, confData.levels, confData.num_term,confData.noiseVariance);
	    MCMCChain<pCN<elasticMixed2dSolver>, elasticMixed2dSolver> plainMCMCSolver(confData.plainMCMC_sample_number, confData.num_term, &singleForwardSolver, confData.pCNstep);
	    plainMCMCSolver.chainInit(&generator);
            plainMCMCSolver.run(&qoimean, &generator);
	    std::cout << "qoi: " << qoimean << std::endl;
            break;
	}


        case 2: //MLMCMC estimation
        {
            double output;
            if (confData.parallelChain == 1){
                MPI_Comm PSubComm;

                int color = rank % (size/confData.levels);
                MPI_Comm_split(MPI_COMM_WORLD, color, rank/confData.levels, &PSubComm);			

                //MLMCMC_Bi_Uniform<pCN<elasticMixed2dSolver>, elasticMixed2dSolver> MLMCMCSolver(PSubComm, confData.levels, confData.num_term, color, confData.a, confData.noiseVariance, confData.randomSeed, 1.0);
                //output = MLMCMCSolver.mlmcmcRun();

                MLMCMC_Bi<pCN<elasticMixed2dSolver>, elasticMixed2dSolver> MLMCMCSolver(PSubComm, confData.levels, confData.num_term, color, confData.a, confData.noiseVariance, confData.randomSeed, confData.pCNstep);
                output = MLMCMCSolver.mlmcmcRun();

                if (rank == color){
                    std::cout << output << std::endl;

                    std::string outputfile = "output_";
                    outputfile.append(std::to_string(rank));

                    std::ofstream myfile;
                    myfile.open(outputfile);
                    for (int i = 0; i<confData.num_term; ++i){
                        myfile << output << " ";
                    }
                    myfile << std::endl;
                    myfile.close();

                }
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0){
                    double buffer;
                    std::string finaloutput = "finalOutput";
                    std::ofstream outputfile;
                    outputfile.open(finaloutput);
                    for (int i = 0; i < size/confData.levels; i++){
                        std::string finalinput = "output_";
                        finalinput.append(std::to_string(i));
                        std::ifstream inputfile;
                        inputfile.open(finalinput, std::ios_base::in);
                        for(int i = 0; i < confData.num_term; ++i){
                            inputfile >> buffer;
                            outputfile << buffer << " ";
                        }
                        outputfile << std::endl;
                        inputfile.close();
                    }
                    outputfile.close();
                }	

            } else {
                //MLMCMC_Bi_Uniform<pCN<elasticMixed2dSolver>, elasticMixed2dSolver> MLMCMCSolver(PETSC_COMM_SELF, confData.levels, 1, rank, confData.a, confData.noiseVariance, confData.randomSeed, 1.0);
                //output = MLMCMCSolver.mlmcmcRun();			

                MLMCMC_Bi<pCN<elasticMixed2dSolver>, elasticMixed2dSolver> MLMCMCSolver(PETSC_COMM_SELF, confData.levels, confData.num_term, (rank+1)*confData.randomSeed/2, confData.a, confData.noiseVariance, confData.randomSeed, confData.pCNstep);
                output = MLMCMCSolver.mlmcmcRun();
                std::cout << output << std::endl;

                std::string outputfile = "output_";
                outputfile.append(std::to_string(rank));

                std::ofstream myfile;
                myfile.open(outputfile);
                for (int i = 0; i<confData.num_term; ++i){
                    myfile << output << " ";
                }
                myfile << std::endl;
                myfile.close();

                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0){
                    double buffer;
                    std::string finaloutput = "finalOutput";
                    std::ofstream outputfile;
                    outputfile.open(finaloutput);
                    for (int i = 0; i < size; i++){
                        std::string finalinput = "output_";
                        finalinput.append(std::to_string(i));
                        std::ifstream inputfile;
                        inputfile.open(finalinput, std::ios_base::in);
                        for(int i = 0; i < 1; ++i){ //confData.num_term; ++i){
                            inputfile >> buffer;
                            outputfile << buffer << " ";
                        }
                        outputfile << std::endl;
                        inputfile.close();
                    }
                    outputfile.close();
                }	
            }
	    break;
        }

        // case 4:
        // {
        //     elasticMixed2dSolver singleForwardSolver(PETSC_COMM_SELF, confData.levels, confData.num_term,confData.noiseVariance);
        //     singleForwardSolver.samples[0]=0.8;
        //     singleForwardSolver.solve(0);

        //     int s=4097;
        //     double x[s],y[s];
        //     double *ux,*uy;
        //     for (int i=0; i<s; i++){x[i]=1.0/(s-1)*i;y[i]=1.0/(s-1)*i;}
        //     ux = (double*) malloc(s*s*sizeof(double));
        //     uy = (double*) malloc(s*s*sizeof(double));

        //     singleForwardSolver.getValues(x, y, ux, uy, s);

        //     Vec X, Workspace;
        //     double H1norm;
        //     structureMesh2D L10Mesh(MPI_COMM_SELF, 10, 3, Q1);
        //     DMCreateGlobalVector(L10Mesh.meshDM, &X);
        //     AssembleA(L10Mesh.A, L10Mesh.meshDM, 1.0);

        //     VortexDOF   **states;
        //     DMDAVecGetArray(L10Mesh.meshDM,X,&states);   

        //     for (int i=0; i<s-1; ++i){
        //         for (int j=0; j<s-1; ++j){
        //             states[j][i].u=ux[s*i+j];
        //             states[j][i].v=uy[s*i+j];
        //             states[j][i].p=0;
        //             std::cout << ux[s*i+j] << " " << uy[s*i+j] << " " << std::endl;
        //         }
        //     }
        //     std::cout << "test" << std::endl;
        //     DMDAVecRestoreArray(L10Mesh.meshDM,X,&states);
        //     std::cout << "test" << std::endl;
        //     VecDuplicate(X, &Workspace);
        //     MatMult(L10Mesh.A,X,Workspace);
        //     VecDot(X,Workspace,&H1norm);

        //     std::cout << H1norm << std::endl;
        //     free(ux);free(uy);
        //     break;            
        // }

        default:
        {
            break;
        }
	}

	PetscFinalize();
	return 0;    
}
