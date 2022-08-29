#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <mpi.h>
#include <petsc.h>
#include <time.h>

#include <confIO.h>
#include <nse2d_Lag.h>
#include <MLMCMC_Bi_Uniform.h>
#include <MLMCMC_Bi.h>

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

	PetscOptionsSetValue(NULL, "-dm_plex_simplex", "1");
	// PetscOptionsSetValue(NULL, "-dm_refine", "1");
	PetscOptionsSetValue(NULL, "-vel_petscspace_degree", "2");
	PetscOptionsSetValue(NULL, "-pres_petscspace_degree", "1");
	PetscOptionsSetValue(NULL, "-ts_type", "beuler");
	PetscOptionsSetValue(NULL, "-ts_max_steps", "10");
	PetscOptionsSetValue(NULL, "-ts_dt", "0.1");
	PetscOptionsSetValue(NULL, "-ts_monitor", NULL);
	PetscOptionsSetValue(NULL, "-dmts_check", NULL);
	PetscOptionsSetValue(NULL, "-snes_monitor_short", NULL);
	PetscOptionsSetValue(NULL, "-snes_converged_reason", NULL);
	PetscOptionsSetValue(NULL, "-ksp_monitor_short", NULL);
	PetscOptionsSetValue(NULL, "-ksp_converged_reason", NULL);
	PetscOptionsSetValue(NULL, "-pc_type", "fieldsplit");
	PetscOptionsSetValue(NULL, "-pc_fieldsplit_type", "schur");
	PetscOptionsSetValue(NULL, "-pc_fieldsplit_schur_fact_type", "full");
	PetscOptionsSetValue(NULL, "-fieldsplit_velocity_pc_type", "hypre");
	PetscOptionsSetValue(NULL, "-fieldsplit_pressure_ksp_rtol", "1.0e-10");
	PetscOptionsSetValue(NULL, "-fieldsplit_pressure_pc_type", "jacobi");

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
            std::cout << "start inverse solver" << std::endl;
            NSE2dSolverLag singleForwardSolver(PETSC_COMM_SELF, confData.levels, confData.num_term,confData.noiseVariance);
            singleForwardSolver.samples[0]=confData.rand_coef[0];
            singleForwardSolver.solve();
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
