#pragma once 

#include <iostream>
#include <math.h>
#include <random>
#include "MLChains_Bi.h"

template <typename samplerType, typename solverType>
class MLMCMC_Bi{
private:
public:
	MPI_Comm PSubComm;
	int levels,sampleSize,M,rank,subRank,a,subSize,baseNum=10;
	double noiseVariance,randomSeed,beta,out;
	double *mean,*L,*A1,*A2,*A3,*A4,*A5,*A6,*A7,*A8;
	std::default_random_engine generator;
	std::vector<std::shared_ptr<solverType>> solvers;

	MLMCMC_Bi(MPI_Comm PSubComm_, int levels_, int sampleSize_, int rank_, int a_, double noiseVariance_, double randomSeed_, double beta_=1): PSubComm(PSubComm_), levels(levels_), sampleSize(sampleSize_), rank(rank_), a(a_), noiseVariance(noiseVariance_), randomSeed(randomSeed_), beta(beta_){
		L = new double[levels+1];
		solvers.resize(levels+1);
		for(int i = 0; i < levels+1; ++i){
			solvers[i] = std::make_shared<solverType>(MPI_COMM_SELF, i, sampleSize, noiseVariance);
		}
		generator.seed(rank_*13+randomSeed_*13.0);
		mean = new double[sampleSize_];
		A1 = new double[sampleSize_];
		A2 = new double[sampleSize_];
		A3 = new double[sampleSize_];
		A4 = new double[sampleSize_];
		A5 = new double[sampleSize_];
		A6 = new double[sampleSize_];
		A7 = new double[sampleSize_];
		A8 = new double[sampleSize_];

		MPI_Comm_rank(PSubComm, &subRank);
		MPI_Comm_size(PSubComm, &subSize);		
		std::cout << "subComm: " << subSize << " " << subRank << " " << rank << std::endl;		
	};
	~MLMCMC_Bi(){
		delete[] L;
		delete[] mean;
		delete[] A1;
		delete[] A2;
		delete[] A3;
		delete[] A4;
		delete[] A5;
		delete[] A6;
		delete[] A7;
		delete[] A8;
	};

	virtual double mlmcmcRun();
    virtual void mlmcmcPreRun(int totalRank, int targetRank);
    virtual double mlmcmcRunInd(int startLocation);
};

template <typename samplerType, typename solverType>
double MLMCMC_Bi<samplerType, solverType>::mlmcmcRun(){
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
            MLChain00<samplerType, solverType> chain00(M, sampleSize, solvers[0].get(), beta);
            chain00.run(mean, &generator);
            L[0] = mean[0];	    		
        }
    } else {
        MLChain00<samplerType, solverType> chain00(M, sampleSize, solvers[0].get(), beta);
        chain00.run(mean, &generator);
        L[0] = mean[0];
    }

    std::cout << rank << " L00: " << L[0] << " " << std::endl;

    if (subSize > 1){
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
        MLChain0i<samplerType, solverType> chain0i(M, sampleSize, solvers[0].get(), solvers[subRank+1].get(), solvers[subRank].get(), beta);
        chain0i.run(mean, &generator);
        L[0] += mean[0];
        std::cout << rank << " L0+: " << L[0] << std::endl;
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
	    M=M*baseNum;
            MLChain0i<samplerType, solverType> chain0i(M, sampleSize, solvers[0].get(), solvers[i].get(), solvers[i-1].get(), beta);
            chain0i.run(mean, &generator);
            L[0] += mean[0];
            std::cout << rank << " L0+: " << L[0] << std::endl;
        }  
        std::cout << rank << " L0 mean: " << L[0] << std::endl;		
        out = L[0];	
    }

    if (subSize > 1){
        if (a == 0){
            M = (int) pow(2, 2*(levels-subRank-1))/pow(levels, 2);
        } else if (a == 2){
            M = (int) pow(2, 2*(levels-subRank-1));
        } else if (a == 3){
            M = (int) pow(levels, 1)*pow(2, 2*(levels-subRank-1));
        } else if (a == 4){
            M = (int) pow(levels, 2)*pow(2, 2*(levels-subRank-1));
        }		
	M = M*baseNum;	
        MLChaini0Upper<samplerType, solverType> chaini0Upper(M, sampleSize, solvers[subRank+1].get(), solvers[subRank].get(), solvers[0].get(), beta);
        chaini0Upper.run(A1, A3, A6, A7, &generator);
        MLChaini0Lower<samplerType, solverType> chaini0Lower(M, sampleSize, solvers[subRank+1].get(), solvers[subRank].get(), solvers[0].get(), beta);
        chaini0Lower.run(A2, A4, A8, A5, &generator);
        L[subRank+1] = A1[0]+A2[0]+A3[0]*(A4[0]+A8[0])+A5[0]*(A6[0]+A7[0]);
        std::cout << rank << " L" << subRank+1 << "+: " << L[subRank+1] << std::endl;
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
	    M=M*baseNum;			
            MLChainijLower<samplerType, solverType> chainijLower(M, sampleSize, solvers[subRank+1].get(), solvers[subRank].get(), solvers[j].get(), solvers[j-1].get(), beta);
            chainijLower.run(A2, A4, A8, A5, &generator);
            MLChainijUpper<samplerType, solverType> chainijUpper(M, sampleSize, solvers[subRank+1].get(), solvers[subRank].get(), solvers[j].get(), solvers[j-1].get(), beta);
            chainijUpper.run(A1, A3, A6, A7, &generator);
            L[subRank+1] += A1[0]+A2[0]+A3[0]*(A4[0]+A8[0])+A5[0]*(A6[0]+A7[0]);
            std::cout << rank << " L" << subRank+1 << "+: " << L[subRank+1] << std::endl;
        }
        MPI_Allreduce(MPI_IN_PLACE, &L[0], levels+1, MPI_DOUBLE, MPI_SUM, PSubComm);
        out = 0;
        for (int i = 0; i < levels+1; ++i){
            if (subRank == 0){
                std::cout << rank << " L" << i << " mean: " << L[i] << std::endl;
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
	    M=M*baseNum;	
            MLChaini0Upper<samplerType, solverType> chaini0Upper(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[0].get(), beta);
            chaini0Upper.run(A1, A3, A6, A7, &generator);
            MLChaini0Lower<samplerType, solverType> chaini0Lower(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[0].get(), beta);
            chaini0Lower.run(A2, A4, A8, A5, &generator);
            L[i] = A1[0]+A2[0]+A3[0]*(A4[0]+A8[0])+A5[0]*(A6[0]+A7[0]);
            std::cout << A1[0] << " " << A2[0] << " " << A3[0] << " " << A4[0]<< " " << A8[0]<< " " << A5[0] << " " << A6[0] << " " << A7[0] << std::endl;
            std::cout << rank << " L" << i << "+: " << L[i] << std::endl;
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
		M=M*baseNum;			
                MLChainijLower<samplerType, solverType> chainijLower(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[j].get(), solvers[j-1].get(), beta);
                chainijLower.run(A2, A4, A8, A5, &generator);
                MLChainijUpper<samplerType, solverType> chainijUpper(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[j].get(), solvers[j-1].get(), beta);
                chainijUpper.run(A1, A3, A6, A7, &generator);
                L[i] += A1[0]+A2[0]+A3[0]*(A4[0]+A8[0])+A5[0]*(A6[0]+A7[0]);
                std::cout << A1[0] << " " << A2[0] << " " << A3[0] << " " << A4[0]<< " " << A8[0]<< " " << A5[0] << " " << A6[0] << " " << A7[0] << std::endl;
                std::cout << rank << " L" << i << "+: " << L[i] << std::endl;
            }
            std::cout << rank << " L" << i << " mean: " << L[i] << std::endl;
            out += L[i];
        }
    }
    return out;
};

template <typename samplerType, typename solverType>
void MLMCMC_Bi<samplerType, solverType>::mlmcmcPreRun(int totalRank, int targetRank){
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
    M = (int)(M*targetRank+10) / targetRank + 1;	    

    MLChain00<samplerType, solverType> chain00(M, sampleSize, solvers[0].get(), beta);
    chain00.generateIndSamples(M, rank, &generator);

    std::cout << " L00 samples generated "<< std::endl;

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
        M = (int)(M*targetRank+10) / targetRank + 1;	    
        MLChain0i<samplerType, solverType> chain0i(M, sampleSize, solvers[0].get(), solvers[i].get(), solvers[i-1].get(), beta);
        chain0i.generateIndSamples(M, rank, i, &generator);
        std::cout <<" L0" << i << " samples generated" << std::endl;
    }  

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
        M = (int)(M*targetRank+10) / targetRank + 1;	    		
        MLChaini0Upper<samplerType, solverType> chaini0Upper(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[0].get(), beta);
        chaini0Upper.generateIndSamples(M,rank,i,&generator);
        MLChaini0Lower<samplerType, solverType> chaini0Lower(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[0].get(), beta);
        chaini0Lower.generateIndSamples(M,rank,i,&generator);
        std::cout << "L" << i << " samples generated" << std::endl;
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
            M = (int)(M*targetRank+10) / targetRank + 1;	    		
            MLChainijLower<samplerType, solverType> chainijLower(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[j].get(), solvers[j-1].get(), beta);
            chainijLower.generateIndSamples(M,rank,i,j,&generator);
            MLChainijUpper<samplerType, solverType> chainijUpper(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[j].get(), solvers[j-1].get(), beta);
            chainijUpper.generateIndSamples(M,rank,i,j,&generator);
            std::cout << "L" << i << j << " samples generated" << std::endl;
        }
        std::cout << "All samples generated" << std::endl;
    }                                   
};


template <typename samplerType, typename solverType>
double MLMCMC_Bi<samplerType, solverType>::mlmcmcRunInd(int startLocation){
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

    for (int i = 0; i < levels+1; ++i){
        L[i] = 0;
    }

    MLChain00<samplerType, solverType> chain00(M, sampleSize, solvers[0].get(), beta);
    chain00.runInd(mean, startLocation*M+1, &generator);
    L[0] = mean[0];

    std::cout << rank << " L00: " << L[0] << " " << std::endl;

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
        MLChain0i<samplerType, solverType> chain0i(M, sampleSize, solvers[0].get(), solvers[i].get(), solvers[i-1].get(), beta);
        chain0i.runInd(mean, startLocation*M+1, i, &generator);
        L[0] += mean[0];
        std::cout << rank << " L0+: " << L[0] << std::endl;
    }  
    std::cout << rank << " L0 mean: " << L[0] << std::endl;		
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
        MLChaini0Upper<samplerType, solverType> chaini0Upper(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[0].get(), beta);
        chaini0Upper.runInd(A1, A3, A6, A7, startLocation*M+1, i, &generator);
        MLChaini0Lower<samplerType, solverType> chaini0Lower(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[0].get(), beta);
        chaini0Lower.runInd(A2, A4, A8, A5, startLocation*M+1, i, &generator);
        L[i] = A1[0]+A2[0]+A3[0]*(A4[0]+A8[0])+A5[0]*(A6[0]+A7[0]);
        std::cout << A1[0] << " " << A2[0] << " " << A3[0] << " " << A4[0]<< " " << A8[0]<< " " << A5[0] << " " << A6[0] << " " << A7[0] << std::endl;
        std::cout << rank << " L" << i << "+: " << L[i] << std::endl;
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
            MLChainijLower<samplerType, solverType> chainijLower(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[j].get(), solvers[j-1].get(), beta);
            chainijLower.runInd(A2, A4, A8, A5, startLocation*M+1, i, j, &generator);
            MLChainijUpper<samplerType, solverType> chainijUpper(M, sampleSize, solvers[i].get(), solvers[i-1].get(), solvers[j].get(), solvers[j-1].get(), beta);
            chainijUpper.runInd(A1, A3, A6, A7, startLocation*M+1, i, j, &generator);
            L[i] += A1[0]+A2[0]+A3[0]*(A4[0]+A8[0])+A5[0]*(A6[0]+A7[0]);
            // std::cout << A1[0] << " " << A2[0] << " " << A3[0] << " " << A4[0]<< " " << A8[0]<< " " << A5[0] << " " << A6[0] << " " << A7[0] << std::endl;
            std::cout << rank << " L" << i << "+: " << L[i] << std::endl;
        }
        std::cout << rank << " L" << i << " mean: " << L[i] << std::endl;
        out += L[i];
    }
    return out;
};
