#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <mpi.h>
#include <petsc.h>

#include <confIO.h>
#include <nse2dCNDir.h>
#include <MLMCMC_Bi_Uniform_Dummy.h>

// static void ref(double x, double y, double time, double output[], double samples[], int sampleSize){
//     output[0] = samples[0]*pow(std::sin(M_PI*x), 2)*std::sin(M_PI*y)*std::cos(M_PI*y)*(exp(time)-1.0);
// 	output[1] = -samples[0]*pow(std::sin(M_PI*y), 2)*std::sin(M_PI*x)*std::cos(M_PI*x)*(exp(time)-1.0);
// }

// static void ref(double x, double y, double time, double output[], double samples[], int sampleSize){
//     output[0] = samples[0]*1000*x*x*pow(1-x,4)*y*y*(1-y)*(3-5*y)*(exp(1)-1);
// 	output[1] = samples[0]*-2000*x*pow(1-x,3)*(1-3*x)*y*y*y*(1-y)*(1-y)*(exp(1)-1);
// }

static void Gref(double x, double y, double time, double output[], double samples[], int sampleSize){
    output[0] = samples[0]*1000*(2*x-12*pow(x,2)+24*pow(x,3)-20*pow(x,4)+6*pow(x,5))*y*y*(1-y)*(3-5*y)*(exp(time)-1);
    output[1] = samples[0]*1000*x*x*pow(1-x,4)*(20*pow(y,3)-24*pow(y,2)+6*y)*(exp(time)-1);
	output[2] = samples[0]*-1000*(30*pow(x,4)-80*pow(x,3)+72*pow(x,2)-24*x+2)*y*y*y*(1-y)*(1-y)*(exp(time)-1);
	output[3] = samples[0]*-2000*x*pow(1-x,3)*(1-3*x)*(3*pow(y,2)-8*pow(y,3)+5*pow(y,4))*(exp(time)-1);
}

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

	// Solver Options List 3
	PetscOptionsSetValue(NULL, "-ksp_type", "gmres");
	PetscOptionsSetValue(NULL, "-pc_fieldsplit_type", "additive");
	PetscOptionsSetValue(NULL, "-fieldsplit_0_pc_type", "ilu");
	PetscOptionsSetValue(NULL, "-fieldsplit_0_ksp_type", "preonly");
	PetscOptionsSetValue(NULL, "-fieldsplit_1_pc_type", "jacobi");
	PetscOptionsSetValue(NULL, "-fieldsplit_1_ksp_type", "preonly");

	// // Solver Options List 4 direct solver mumps
	// PetscOptionsSetValue(NULL, "-ksp_type", "preonly");
	// PetscOptionsSetValue(NULL, "-pc_type", "lu");
	// PetscOptionsSetValue(NULL, "-pc_factor_mat_solver_type", "mumps");
	// PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "2");
	// PetscOptionsSetValue(NULL, "-mat_mumps_icntl_24", "1");
	// PetscOptionsSetValue(NULL, "-mat_mumps_cntl_3", "1e-6");

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
            NSE2dSolverCNDir singleForwardSolver(PETSC_COMM_SELF, confData.levels, confData.num_term,confData.noiseVariance, DM_BOUNDARY_NONE);
            singleForwardSolver.samples[0]=confData.rand_coef[0];
            singleForwardSolver.solve();
            singleForwardSolver.lnLikelihood();
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

                MLMCMC_Bi_Uniform<pCN<NSE2dSolverCNDir>, NSE2dSolverCNDir> MLMCMCSolver(PSubComm, confData.levels, 1, color, confData.a, confData.noiseVariance, confData.randomSeed, 1.0);
                output = MLMCMCSolver.mlmcmcRun();

                // MLMCMC_Bi<pCN<stokesSolver>, stokesSolver> MLMCMCSolver(PSubComm, levels, 1, (rank+randomSeed)*randomSeed, a, noiseVariance, 1.0);
                // output = MLMCMCSolver.mlmcmcRun();

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
                MLMCMC_Bi_Uniform<pCN<NSE2dSolverCNDir>, NSE2dSolverCNDir> MLMCMCSolver(PETSC_COMM_SELF, confData.levels, 1, rank, confData.a, confData.noiseVariance, confData.randomSeed, 1.0);
                output = MLMCMCSolver.mlmcmcRun();			

                // MLMCMC_Bi<pCN<stokesSolver>, stokesSolver> MLMCMCSolver(PETSC_COMM_SELF, levels, 1, rank, a, noiseVariance, 1.0);
                // output = MLMCMCSolver.mlmcmcRun();
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
            break;
        }

        // case 3:
        // {
        //     NSE2dDirSolver singleForwardSolver(PETSC_COMM_SELF, confData.levels, confData.num_term,confData.noiseVariance, DM_BOUNDARY_NONE);
        //     singleForwardSolver.samples[0]=0.8;
        //     singleForwardSolver.solve();

        //     int s=2049;
        //     double x[s],y[s];
        //     double *ux,*uy;
        //     for (int i=0; i<s; i++){x[i]=1.0/(s-1)*i;y[i]=1.0/(s-1)*i;}
        //     ux = (double*) malloc(s*s*sizeof(double));
        //     uy = (double*) malloc(s*s*sizeof(double));

        //     singleForwardSolver.getValues(x, y, ux, uy, s);

        //     Vec load,reference,solution,workspace;
        //     double H1norm;
        //     structureMesh2D L10Mesh(MPI_COMM_SELF, 9, 3, Q1, DM_BOUNDARY_NONE);
        //     AssembleM(L10Mesh.M, L10Mesh.meshDM);

        //     DMCreateGlobalVector(L10Mesh.meshDM, &load);
        //     DMCreateGlobalVector(L10Mesh.meshDM, &reference);
        //     VecDuplicate(reference,&solution);
        //     VecZeroEntries(load);
        //     AssembleF(load,L10Mesh.meshDM,1.0,ref,singleForwardSolver.samples.get(),singleForwardSolver.num_term);

        //     MatZeroRowsColumnsIS(L10Mesh.M,L10Mesh.bottomUx,0.0,reference,load);
        //     MatZeroRowsColumnsIS(L10Mesh.M,L10Mesh.bottomUy,0.0,reference,load);
        //     MatZeroRowsColumnsIS(L10Mesh.M,L10Mesh.leftUx,0.0,reference,load);
        //     MatZeroRowsColumnsIS(L10Mesh.M,L10Mesh.leftUy,0.0,reference,load);
        //     MatZeroRowsColumnsIS(L10Mesh.M,L10Mesh.rightUx,0.0,reference,load);
        //     MatZeroRowsColumnsIS(L10Mesh.M,L10Mesh.rightUy,0.0,reference,load);
        //     MatZeroRowsColumnsIS(L10Mesh.M,L10Mesh.topUx,0.0,reference,load);
        //     MatZeroRowsColumnsIS(L10Mesh.M,L10Mesh.topUy,0.0,reference,load);


        //     Interpolate(L10Mesh.M, load, reference);
        //     // VecView(reference, PETSC_VIEWER_STDOUT_WORLD);

        //     std::cout << "interpolate solution" << std::endl;
        //     VortexDOF   **states;
        //     DMDAVecGetArray(L10Mesh.meshDM,solution,&states);   
        //     for (int i=0; i<s-1; ++i){
        //         for (int j=0; j<s-1; ++j){
        //             states[i][j].u=ux[s*i+j];
        //             states[i][j].v=uy[s*i+j];
        //             states[i][j].p=0;
        //         }
        //     }
        //     DMDAVecRestoreArray(L10Mesh.meshDM,solution,&states);
        //     // VecView(solution, PETSC_VIEWER_STDOUT_WORLD);

        //     std::cout << "compute H1norm" << std::endl;
        //     VecAXPY(solution,-1.0,reference);
        //     // VecView(solution, PETSC_VIEWER_STDOUT_WORLD);

        //     VecDuplicate(solution,&workspace);
        //     AssembleA(L10Mesh.A, L10Mesh.meshDM, 1.0);
        //     MatMult(L10Mesh.A,solution,workspace);
        //     VecDot(solution,solution,&H1norm);

        //     std::cout << H1norm << std::endl;
        //     VecDestroy(&load);
        //     VecDestroy(&reference);
        //     VecDestroy(&solution);
        //     VecDestroy(&workspace);

        //     free(ux);free(uy);
        //     break;            
        // }

        case 4:
        {
            NSE2dSolverCNDir singleForwardSolver(PETSC_COMM_SELF, confData.levels, confData.num_term,confData.noiseVariance, DM_BOUNDARY_NONE);
            singleForwardSolver.samples[0]=0.8;
            singleForwardSolver.solve();

            std::cout << "compute H1" << std::endl;

            int ngp=9;
            int meshsize=512;
            double quadx[ngp];
            double quadw[ngp];
            double x[ngp],y[ngp];
            double sum=0,output[4];//],output2[4];
            gauleg(ngp, quadx, quadw);  //gaussian legendre
            for (int i=0; i<ngp; i++){quadx[i]=(quadx[i]+1)/meshsize/2;quadw[i]=quadw[i]/meshsize/2;}
            
            double *uxx, *uxy, *uyx, *uyy;
            uxx = (double*) malloc(ngp*ngp*sizeof(double));
            uxy = (double*) malloc(ngp*ngp*sizeof(double));
            uyx = (double*) malloc(ngp*ngp*sizeof(double));
            uyy = (double*) malloc(ngp*ngp*sizeof(double));

            for (int i=0; i<meshsize; ++i){
                for (int j=0; j<meshsize; ++j){
                    for (int n=0; n<ngp; n++)
                    {
                        x[n]=quadx[n]+(double)i/meshsize;
                        y[n]=quadx[n]+(double)j/meshsize;
                    }            
                    singleForwardSolver.getGradients(x,y,uxx,uxy,uyx,uyy,ngp);
                    for (int n=0; n<ngp; n++){
                        for (int m=0; m<ngp; m++){
                            Gref(x[m], y[n], 1.0, output, singleForwardSolver.samples.get(), singleForwardSolver.num_term);
                            uxx[n*ngp+m] = uxx[n*ngp+m]-output[0];
                            uxy[n*ngp+m] = uxy[n*ngp+m]-output[1];
                            uyx[n*ngp+m] = uyx[n*ngp+m]-output[2];
                            uyy[n*ngp+m] = uyy[n*ngp+m]-output[3];

                            // std::cout << uxx[n*ngp+m] << " " << output[0] << " " << quadw[m]*quadw[n] << " "  << std::endl;

                            sum += quadw[m]*quadw[n]*pow(uxx[n*ngp+m],2);
                            sum += quadw[m]*quadw[n]*pow(uxy[n*ngp+m],2);
                            sum += quadw[m]*quadw[n]*pow(uyx[n*ngp+m],2);
                            sum += quadw[m]*quadw[n]*pow(uyy[n*ngp+m],2);
                        }
                    }                                
                }
            }
            std::cout << "H1 norm: "<< sum << std::endl;

            free(uxx);free(uxy);free(uyx);free(uyy);
            break;            
        }

        case 8:
        {
            MLMCMC_Bi_Uniform<pCN<NSE2dSolverCNDir>, NSE2dSolverCNDir> MLMCMCSolver(PETSC_COMM_SELF, confData.levels, 1, rank, confData.a, confData.noiseVariance, confData.randomSeed, 1.0);
            MLMCMCSolver.mlmcmcPreRun(size, size);
            break;
        }

        case 9:
        {
            double output;
            MLMCMC_Bi_Uniform<pCN<NSE2dSolverCNDir>, NSE2dSolverCNDir> MLMCMCSolver(PETSC_COMM_SELF, confData.levels, 1, rank, confData.a, confData.noiseVariance, confData.randomSeed, 1.0);
            std::cout << output << std::endl;
            std::string finaloutput = "finalOutput";
            std::ofstream outputfile;
            outputfile.open(finaloutput);
            for (int i=0; i<8; ++i){
                output=MLMCMCSolver.mlmcmcRunInd(i);
                std::cout << output << std::endl;
                outputfile << output << std::endl; 
            }        
            outputfile.close();   
            break;            
        }

        default:
        {
            break;
        }
	}



	PetscFinalize();
	return 0;    
}