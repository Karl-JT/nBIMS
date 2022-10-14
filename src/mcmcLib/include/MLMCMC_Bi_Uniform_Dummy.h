#pragma once 

#include <iostream>
#include <math.h>
#include <random>
#include "MLChains_Bi_Uniform.h"

template <typename samplerType, typename solverType>
class MLMCMC_Bi_Uniform{
private:
public:
	MPI_Comm PSubComm;
	int levels;
	int sampleSize;
	int M;
	int color;
	int a;
	int subRank;
	int subSize;
	double noiseVariance,randomSeed,beta,baseNum=1;
	double* mean;
	double out;
	double* L;
	double* A0;
	double* A1;
	double* A2;
	std::default_random_engine generator;
	std::vector<std::shared_ptr<solverType>> solvers;

	MLMCMC_Bi_Uniform(MPI_Comm PSubComm_, int levels_, int sampleSize_, int color_, int a_, double noiseVariance_, double randomSeed_, int beta_=1): PSubComm(PSubComm_), levels(levels_), sampleSize(sampleSize_), color(color_), a(a_), noiseVariance(noiseVariance_), randomSeed(randomSeed_), beta(beta_){
		L = new double[levels+1];
		solvers.resize(levels+1);
		for(int i = 0; i < levels+1; ++i){
			solvers[i] = std::make_shared<solverType>(MPI_COMM_SELF, i, sampleSize, noiseVariance);
            solvers[i]->rank = color;
		}
		generator.seed(color*13.0+randomSeed_*13.0);
		mean = new double[sampleSize_];
		A0 = new double[sampleSize_];
		A1 = new double[sampleSize_];
		A2 = new double[sampleSize_];

		MPI_Comm_rank(PSubComm, &subRank);
		MPI_Comm_size(PSubComm, &subSize);		
		std::cout << "subComm: " << subSize << " " << subRank << " " << color << std::endl;
	};
	~MLMCMC_Bi_Uniform(){
		delete[] L;
		delete[] mean;
		delete[] A0;
		delete[] A1;
		delete[] A2;
	};

    virtual double mlmcmcRun();
    virtual void mlmcmcPreRun(int totalRank, int targetRank);
    virtual double mlmcmcRunInd(int startLocation);
};

template <typename samplerType, typename solverType>
double MLMCMC_Bi_Uniform<samplerType, solverType>::mlmcmcRun(){
    if (a == 0){
        M = (int) pow(2.0, 2.0*levels)/pow(levels, 4);
    } else if (a == 2){
        M = (int) pow(2.0, 2.0*levels)/pow(levels, 2);
    } else if (a == 3){
        M = (int) pow(2.0, 2.0*levels)/pow(levels, 1);    	
    } else if (a == 4){
        M = (int) pow(2.0, 2.0*levels)/pow(log(levels), 2);    	
    }		
    M = (int) std::max(1, M);		    
    M = M*baseNum;

    for (int i = 0; i < levels+1; ++i){
        L[i] = 0;
    }
    if (subSize > 1){
        if (subRank == 0){
            MLChain_Uni_00<samplerType, solverType> chain00(M, sampleSize, solvers[0].get());
            chain00.run(mean, &generator);
            L[0] = mean[0];	    	
            std::cout << "color: " << color << " subRank: " << subRank << " L00: " << L[0] << std::endl;
        }
    } else {
        MLChain_Uni_00<samplerType, solverType> chain00(M, sampleSize, solvers[0].get());
        chain00.run(mean, &generator);
        L[0] = mean[0];
        std::cout << "color: " << color << " subrank: " << subRank << " L00: " << L[0] << std::endl;
    }

    if (subSize > 1){
        if (subRank < levels){//temprary
            if (a == 0){
                M = (int) pow(2, 2*(levels-subRank-1))/pow(levels, 2);
            } else if (a == 2){
                M = (int) pow(2, 2*(levels-subRank-1));
            } else if (a == 3){
                M = (int) pow(levels, 1)*pow(2, 2*(levels-subRank-1));
            } else if (a == 4){
                M = (int) pow(levels, 2)*pow(2, 2*(levels-subRank-1));
            }			
            M = (int) std::max(1, M);
            M = M*baseNum;

            MLChain_Uni_0i<samplerType, solverType> chain0i(M, sampleSize, solvers[0].get(), solvers[subRank+1].get(), solvers[subRank].get());
            chain0i.run(mean, &generator);
            L[0] += mean[0];		 

            std::cout << "color: " << color << " subRank: " << subRank << " L0+: " << L[0] << " forward solver: " << subRank+1 << std::endl;
        }
    } else {
        for (int i = 1; i < levels+1; ++i){
            if (a == 0){
                M = (int) pow(2, 2*(levels-i))/pow(levels, 2);
            } else if (a == 2){
                M = (int) pow(2, 2*(levels-i));
            } else if (a == 3){
                M = (int) pow(levels, 1)*pow(2, 2*(levels-i));
            } else if (a == 4){
                M = (int) pow(levels, 2)*pow(2, 2*(levels-i));
            }
            M = (int) std::max(1, M);
            M = M*baseNum;
            MLChain_Uni_0i<samplerType, solverType> chain0i(M, sampleSize, solvers[0].get(), solvers[i].get(), solvers[i-1].get());
            chain0i.run(mean, &generator);
            L[0] += mean[0];
            std::cout << "color: " << color << " L0+: " << L[0] << std::endl;
        }  	
        std::cout << "color: " << color << " L0 mean: " << L[0] << std::endl;
        out = L[0];
    }

    if (subSize > 1){
        if (subRank < levels){//temprary
            if (a == 0){
                M = (int) pow(2, 2*(levels-subRank-1))/pow(levels, 2);
            } else if (a == 2){
                M = (int) pow(2, 2*(levels-subRank-1));
            } else if (a == 3){
                M = (int) pow(levels, 1)*pow(2, 2*(levels-subRank-1));
            } else if (a == 4){
                M = (int) pow(levels, 2)*pow(2, 2*(levels-subRank-1));
            }
            M = (int) std::max(1, M);
            M = M*baseNum;

            MLChain_Uni_i0Upper<samplerType, solverType> chaini0Upper(M, sampleSize, solvers[subRank+1].get(), solvers[subRank].get(), solvers[0].get());		
            chaini0Upper.run(A1, A2, &generator);
            std::cout << "subRank: " << subRank << " A1, A2 completed" << std::endl;

            MLChain_Uni_i0Lower<samplerType, solverType> chaini0Lower(M, sampleSize, solvers[subRank].get(), solvers[0].get());
            chaini0Lower.run(A0, &generator);
            std::cout << "subRank: " << subRank << " A0 completed" << std::endl;


            L[subRank+1] = A1[0]+A0[0]*A2[0];
            std::cout << "color " << color << " L" << subRank+1 << "+: " << L[subRank+1] << std::endl;
            // std::cout << A1[0] << " " << A0[0] << " " <<  A2[0] << std::endl;
            for (int j = 1; j < levels-subRank; ++j){
                if (a == 0){
                    M = (int) pow(2, 2*(levels-subRank-1-j));
                } else if (a == 2){
                    M = (int) pow(subRank+1+j, 2)*pow(2, 2*(levels-subRank-1-j));
                } else if (a == 3){
                    M = (int) pow(subRank+1+j, 3)*pow(2, 2*(levels-subRank-1-j));
                } else if (a == 4){
                    M = (int) pow(subRank+1+j, 4)*pow(2, 2*(levels-subRank-1-j));
                }
                M = (int) std::max(1, M);
                M = M*baseNum;

                MLChain_Uni_ijLower<samplerType, solverType> chainijLower(M, sampleSize, solvers[subRank].get(), solvers[j].get(), solvers[j-1].get());
                chainijLower.run(A0, &generator);

                MLChain_Uni_ijUpper<samplerType, solverType> chainijUpper(M, sampleSize, solvers[subRank+1].get(), solvers[subRank].get(), solvers[j].get(), solvers[j-1].get());
                chainijUpper.run(A1, A2, &generator);
                L[subRank+1] += A1[0]+ A0[0]*A2[0];

                std::cout << "color: " << color << " L" << subRank+1 << "+: " << L[subRank+1] << std::endl;
                // std::cout << A1[0] << " " << A0[0] << " " <<  A2[0] << std::endl;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &L[0], levels+1, MPI_DOUBLE, MPI_SUM, PSubComm);
        out = 0;
        for (int i = 0; i < levels+1; ++i){
            if (subRank == 0){
                std::cout << "color: " << color << " L" << i << " mean: " << L[i] << std::endl;
            }		    	
            out += L[i];
        }			
    } else {
        for (int i = 1; i < levels+1; ++i){
            if (a == 0){
                M = (int) pow(2, 2*(levels-i))/pow(levels, 2);
            } else if (a == 2){
                M = (int) pow(2, 2*(levels-i));
            } else if (a == 3){
                M = (int) pow(levels, 1)*pow(2, 2*(levels-i));
            } else if (a == 4){
                M = (int) pow(levels, 2)*pow(2, 2*(levels-i));
            }
            M = (int) std::max(1, M);
            M = M*baseNum;

            MLChain_Uni_i0Upper<samplerType, solverType> chaini0Upper(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[0].get());
            chaini0Upper.run(A1, A2, &generator);
            MLChain_Uni_i0Lower<samplerType, solverType> chaini0Lower(M, sampleSize, solvers[i-1].get(), solvers[0].get());
            chaini0Lower.run(A0, &generator);
            L[i] = A1[0]+A0[0]*A2[0];
            std::cout << A1[0] << " " << A0[0] << " " <<  A2[0] << std::endl;
            std::cout << "color: " << color << " L" << i << "+: " << L[i] << std::endl;
            for (int j = 1; j < levels-i+1; ++j){
                if (a == 0){
                    M = (int) pow(2, 2*(levels-i-j));
                } else if (a == 2){
                    M = (int) pow(i+j, 2)*pow(2, 2*(levels-i-j));
                } else if (a == 3){
                    M = (int) pow(i+j, 3)*pow(2, 2*(levels-i-j));
                } else if (a == 4){
                    M = (int) pow(i+j, 4)*pow(2, 2*(levels-i-j));
                }
                M = (int) std::max(1, M);
                M = M*baseNum;

                MLChain_Uni_ijLower<samplerType, solverType> chainijLower(M, sampleSize, solvers[i-1].get(), solvers[j].get(), solvers[j-1].get());
                chainijLower.run(A0, &generator);
                MLChain_Uni_ijUpper<samplerType, solverType> chainijUpper(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[j].get(), solvers[j-1].get());
                chainijUpper.run(A1, A2, &generator);
                L[i] += A1[0]+ A0[0]*A2[0];
                std::cout << A1[0] << " " << A0[0] << " " <<  A2[0] << std::endl;
                std::cout << "color: " << color << " L" << i << "+: " << L[i] << std::endl;
            }
            std::cout << "color: " << color << " L" << i << " mean: " << L[i] << std::endl;
            out += L[i];
        }
    }
    return out;
};


template <typename samplerType, typename solverType>
void MLMCMC_Bi_Uniform<samplerType, solverType>::mlmcmcPreRun(int totalRank, int targetRank)
{
    if (a == 0){
        M = (int) pow(2.0, 2.0*levels)/pow(levels, 4);
    } else if (a == 2){
        M = (int) pow(2.0, 2.0*levels)/pow(levels, 2);
    } else if (a == 3){
        M = (int) pow(2.0, 2.0*levels)/pow(levels, 1);    	
    } else if (a == 4){
        M = (int) pow(2.0, 2.0*levels)/pow(log(levels), 2);    	
    }		
    M = (int) std::max(1, M);		    
    M = (int)(M+5)*targetRank / totalRank + 1;	
    M = M*baseNum;

    MLChain_Uni_00<samplerType, solverType> chain00(M, sampleSize, solvers[0].get());
    chain00.generateIndSamples(M, color, &generator);
    std::cout << "L00 samples generated " << std::endl;

    for (int i = 1; i < levels+1; ++i){
        if (a == 0){
            M = (int) pow(2, 2*(levels-i))/pow(levels, 2);
        } else if (a == 2){
            M = (int) pow(2, 2*(levels-i));
        } else if (a == 3){
            M = (int) pow(levels, 1)*pow(2, 2*(levels-i));
        } else if (a == 4){
            M = (int) pow(levels, 2)*pow(2, 2*(levels-i));
        }
        M = (int) std::max(1, M);
        M = (int)(M+5)*targetRank / totalRank + 1;	
        M = M*baseNum;

        MLChain_Uni_0i<samplerType, solverType> chain0i(M, sampleSize, solvers[0].get(), solvers[i].get(), solvers[i-1].get());
        chain0i.generateIndSamples(M, color, i, &generator);
    }  	
    std::cout << "L0 samples generated " << std::endl;

    for (int i = 1; i < levels+1; ++i){
        if (a == 0){
            M = (int) pow(2, 2*(levels-i))/pow(levels, 2);
        } else if (a == 2){
            M = (int) pow(2, 2*(levels-i));
        } else if (a == 3){
            M = (int) pow(levels, 1)*pow(2, 2*(levels-i));
        } else if (a == 4){
            M = (int) pow(levels, 2)*pow(2, 2*(levels-i));
        }
        M = (int)(M+5)*targetRank / totalRank + 1;
        M = M*baseNum;
	
        MLChain_Uni_i0Upper<samplerType, solverType> chaini0Upper(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[0].get());
        chaini0Upper.generateIndSamples(M, color, i, &generator);
        MLChain_Uni_i0Lower<samplerType, solverType> chaini0Lower(M, sampleSize, solvers[i-1].get(), solvers[0].get());
        chaini0Lower.generateIndSamples(M, color, i, &generator);

        for (int j = 1; j < levels-i+1; ++j){
            if (a == 0){
                M = (int) pow(2, 2*(levels-i-j));
            } else if (a == 2){
                M = (int) pow(i+j, 2)*pow(2, 2*(levels-i-j));
            } else if (a == 3){
                M = (int) pow(i+j, 3)*pow(2, 2*(levels-i-j));
            } else if (a == 4){
                M = (int) pow(i+j, 4)*pow(2, 2*(levels-i-j));
            }
            M = (int)(M+5)*targetRank / totalRank + 1;	
            M = M*baseNum;
            MLChain_Uni_ijLower<samplerType, solverType> chainijLower(M, sampleSize, solvers[i-1].get(), solvers[j].get(), solvers[j-1].get());
            chainijLower.generateIndSamples(M, color, i, j, &generator);
            MLChain_Uni_ijUpper<samplerType, solverType> chainijUpper(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[j].get(), solvers[j-1].get());
            chainijUpper.generateIndSamples(M, color, i, j, &generator);
        }
        std::cout << " L" << i << " samples generated " << std::endl;
    }  
};

template <typename samplerType, typename solverType>
double MLMCMC_Bi_Uniform<samplerType, solverType>::mlmcmcRunInd(int startLocation)
{
    if (a == 0){
        M = (int) pow(2.0, 2.0*levels)/pow(levels, 4);
    } else if (a == 2){
        M = (int) pow(2.0, 2.0*levels)/pow(levels, 2);
    } else if (a == 3){
        M = (int) pow(2.0, 2.0*levels)/pow(levels, 1);    	
    } else if (a == 4){
        M = (int) pow(2.0, 2.0*levels)/pow(log(levels), 2);    	
    }		
    M = (int) std::max(1, M);
    M = M*baseNum;		    

    for (int i = 0; i < levels+1; ++i){
        L[i] = 0;
    }

    MLChain_Uni_00<samplerType, solverType> chain00(M, sampleSize, solvers[0].get());
    chain00.runInd(mean, startLocation*(M+5)+1, &generator);
    L[0] = mean[0];
    std::cout << "color: " << color << " subrank: " << subRank << " L00: " << L[0] << std::endl;

    for (int i = 1; i < levels+1; ++i){
        if (a == 0){
            M = (int) pow(2, 2*(levels-i))/pow(levels, 2);
        } else if (a == 2){
            M = (int) pow(2, 2*(levels-i));
        } else if (a == 3){
            M = (int) pow(levels, 1)*pow(2, 2*(levels-i));
        } else if (a == 4){
            M = (int) pow(levels, 2)*pow(2, 2*(levels-i));
        }
        M = (int) std::max(1, M);
        M = M*baseNum;
        MLChain_Uni_0i<samplerType, solverType> chain0i(M, sampleSize, solvers[0].get(), solvers[i].get(), solvers[i-1].get());
        chain0i.runInd(mean,startLocation*(M+5)+1, i, &generator);
        L[0] += mean[0];
        std::cout << "color: " << color << " L0+: " << L[0] << std::endl;
    }  	
    std::cout << "color: " << color << " L0 mean: " << L[0] << std::endl;
    out = L[0];

    for (int i = 1; i < levels+1; ++i){
        if (a == 0){
            M = (int) pow(2, 2*(levels-i))/pow(levels, 2);
        } else if (a == 2){
            M = (int) pow(2, 2*(levels-i));
        } else if (a == 3){
            M = (int) pow(levels, 1)*pow(2, 2*(levels-i));
        } else if (a == 4){
            M = (int) pow(levels, 2)*pow(2, 2*(levels-i));
        }
        M = (int) std::max(1, M);
        M = M*baseNum;

        MLChain_Uni_i0Upper<samplerType, solverType> chaini0Upper(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[0].get());
        chaini0Upper.runInd(A1, A2, startLocation*(M+5)+1, i, &generator);
        MLChain_Uni_i0Lower<samplerType, solverType> chaini0Lower(M, sampleSize, solvers[i-1].get(), solvers[0].get());
        chaini0Lower.runInd(A0, startLocation*(M+5)+1, i, &generator);
        L[i] = A1[0]+A0[0]*A2[0];
        std::cout << A1[0] << " " << A0[0] << " " <<  A2[0] << std::endl;
        std::cout << "color: " << color << " L" << i << "+: " << L[i] << std::endl;
        for (int j = 1; j < levels-i+1; ++j){
            if (a == 0){
                M = (int) pow(2, 2*(levels-i-j));
            } else if (a == 2){
                M = (int) pow(i+j, 2)*pow(2, 2*(levels-i-j));
            } else if (a == 3){
                M = (int) pow(i+j, 3)*pow(2, 2*(levels-i-j));
            } else if (a == 4){
                M = (int) pow(i+j, 4)*pow(2, 2*(levels-i-j));
            }
            M = (int) std::max(1, M);
            M = M*baseNum;

            MLChain_Uni_ijLower<samplerType, solverType> chainijLower(M, sampleSize, solvers[i-1].get(), solvers[j].get(), solvers[j-1].get());
            chainijLower.runInd(A0, startLocation*(M+5)+1, i, j, &generator);
            MLChain_Uni_ijUpper<samplerType, solverType> chainijUpper(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[j].get(), solvers[j-1].get());
            chainijUpper.runInd(A1, A2, startLocation*(M+5)+1, i, j, &generator);
            L[i] += A1[0]+ A0[0]*A2[0];
            std::cout << A1[0] << " " << A0[0] << " " <<  A2[0] << std::endl;
            std::cout << "color: " << color << " L" << i << "+: " << L[i] << std::endl;
        }
        std::cout << "color: " << color << " L" << i << " mean: " << L[i] << std::endl;
        out += L[i];
    }

    return out;    
}
