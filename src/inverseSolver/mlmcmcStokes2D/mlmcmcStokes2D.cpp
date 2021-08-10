#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <mpi.h>
#include <petsc.h>

#include <confIO.h>
#include <stokes2d.h>
//#include <MLMCMC_Bi_Uniform_Dummy.h>
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

	// // Solver Options List 2
	// PetscOptionsSetValue(NULL, "-ksp_type", "gmres");
	// PetscOptionsSetValue(NULL, "-fieldsplit_0_ksp_type", "preonly");
	// PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_type", "lu");
	// PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_factor_mat_solver_type", "mumps");
	// PetscOptionsSetValue(NULL, "-fieldsplit_1_ksp_type", "preonly");

	// // Solver Options List 3
	// PetscOptionsSetValue(NULL, "-ksp_type", "gmres");
	// PetscOptionsSetValue(NULL, "-pc_fieldsplit_type", "additive");
	// PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_type", "gamg");
	// PetscOptionsSetValue(NULL, "-fieldsplit_0_ksp_type", "preonly");
	// PetscOptionsSetValue(NULL, "-fieldsplit_1_pc_type", "jacobi");
	// PetscOptionsSetValue(NULL, "-fieldsplit_1_ksp_type", "preonly");

	// Solver Options List 4 direct solver mumps
	PetscOptionsSetValue(NULL, "-ksp_type", "preonly");
	PetscOptionsSetValue(NULL, "-pc_type", "cholesky");
	PetscOptionsSetValue(NULL, "-pc_factor_mat_solver_type", "mumps");
	PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "2");
	PetscOptionsSetValue(NULL, "-mat_mumps_icntl_24", "1");
	PetscOptionsSetValue(NULL, "-mat_mumps_cntl_3", "1e-6");

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
            stokes2dSolver singleForwardSolver(PETSC_COMM_SELF, confData.levels, confData.num_term,confData.noiseVariance);
            singleForwardSolver.samples[0]=confData.rand_coef[0];
            singleForwardSolver.solve();
            std::cout << singleForwardSolver.ObsOutput() << std::endl;
            std::cout << singleForwardSolver.QoiOutput() << std::endl;
            break;
        }

        case 2: //MLMCMC estimation
        {
            double output;
            if (confData.parallelChain == 1){
                MPI_Comm PSubComm;

                int color = rank % (size/confData.levels);
                // int color = rank % (size/(levels+1)); //temprary
                MPI_Comm_split(PETSC_COMM_WORLD, color, rank/confData.levels, &PSubComm);			
                // MPI_Comm_split(PETSC_COMM_WORLD, color, rank/(levels+1), &PSubComm); //temprary			

                // MLMCMC_Bi_Uniform<pCN<stokes2dSolver>, stokes2dSolver> MLMCMCSolver(PSubComm, confData.levels, confData.num_term, color, confData.a, confData.noiseVariance, confData.randomSeed, 1.0);
                // output = MLMCMCSolver.mlmcmcRun();

                MLMCMC_Bi<pCN<stokes2dSolver>, stokes2dSolver> MLMCMCSolver(PSubComm, confData.levels, confData.num_term, color, confData.a, confData.noiseVariance, confData.randomSeed, confData.pCNstep);
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
                // MLMCMC_Bi_Uniform<pCN<stokes2dSolver>, stokes2dSolver> MLMCMCSolver(PETSC_COMM_SELF, confData.levels, 1, rank, confData.a, confData.noiseVariance, confData.randomSeed, 1.0);
                // output = MLMCMCSolver.mlmcmcRun();			

                MLMCMC_Bi<pCN<stokes2dSolver>, stokes2dSolver> MLMCMCSolver(PETSC_COMM_SELF, confData.levels, confData.num_term, (rank+1)*confData.randomSeed, confData.a, confData.noiseVariance, confData.pCNstep);
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
                        for(int i = 0; i < confData.num_term; ++i){
                            inputfile >> buffer;
                            outputfile << buffer << " ";
                        }
                        outputfile << std::endl;
                        inputfile.close();
                    }
                    outputfile.close();
                }	
            }
        }

        // case 4:
        // {
        //     stokes2dSolver singleForwardSolver(PETSC_COMM_SELF, confData.levels, confData.num_term,confData.noiseVariance);
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