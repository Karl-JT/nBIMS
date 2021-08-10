#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include "MCMCChain.h"

template <typename samplerType, typename solverType> 
class MLChain00 : public MCMCChain<samplerType, solverType> {
public:
	MLChain00(int maxChainLength_, int sampleSize_, solverType* solver_, double beta_=1.0) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_, beta_){};
	~MLChain00(){};

    virtual void generateIndSamples(int targetSampleSize, int rank, std::default_random_engine* generator);
    virtual void runInd(double QoImean[], int startLocation, std::default_random_engine* generator);
};

template <typename samplerType, typename solverType> 
void MLChain00<samplerType, solverType>::generateIndSamples(int targetSampleSize, int rank, std::default_random_engine* generator)
{
    std::string outputfile = "chain00_ind_samples_";
    outputfile.append(std::to_string(rank));

    std::ofstream myfile;
    myfile.open(outputfile, std::ios_base::app);

    myfile<<"Obs,Likelihood,QoI,samples"<<std::endl;
    double Workspace;
	double initialSample[this->sampleSize];
	this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));
	this->solver->priorSample(initialSample);
	for (int i = 0; i < this->sampleSize; ++i){
		this->solver->samples[i] = initialSample[i];
	}
	this->solver->solve();
	Workspace=this->solver->solve4Obs();myfile<<Workspace;
    Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
    Workspace=this->solver->solve4QoI();myfile<<","<<Workspace;

    for (int i = 0; i<this->sampleSize; ++i){
        Workspace=this->solver->samples[i];myfile<<","<<Workspace;
    }
    myfile << std::endl;

	for (int i = 1; i < targetSampleSize; ++i){
        this->sampleProposal();
        this->solver->solve();
        Workspace=this->solver->solve4Obs();myfile<<Workspace;
        Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
        Workspace=this->solver->solve4QoI();myfile<<","<<Workspace;

        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<","<<Workspace;
        }
        myfile << std::endl;
	}
    myfile.close();
} 


template <typename samplerType, typename solverType> 
void MLChain00<samplerType, solverType>::runInd(double QoImean[], int startLocation, std::default_random_engine* generator){
    std::ifstream indSamples("chain00_ind_samples");
    std::string line;
    std::string temp;
    getline(indSamples,line);

    //go to position
    for (int i=0; i<startLocation;++i)
    {
        getline(indSamples,line);
    }
    std::stringstream ss(line);
    getline(ss,temp,',');
    // double ObsInd=std::stod(temp);
    getline(ss,temp,',');
    double Likelihoodt0Ind=std::stod(temp);
    double Likelihoodt1Ind=Likelihoodt0Ind;
    getline(ss,temp,',');
    double QoiIndt0=std::stod(temp);
    double QoiIndt1=QoiIndt0;

    this->chainIdx=1;
    if (this->chainIdx < this->numBurnin)
    {
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);
        getline(ss,temp,',');
        // ObsInd=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1Ind=std::stod(temp);
        getline(ss,temp,',');
        QoiIndt1=std::stod(temp);


        this->alpha = this->sampler->getAlpha(Likelihoodt0Ind, Likelihoodt1Ind);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0Ind=Likelihoodt1Ind;
            QoiIndt0=QoiIndt1;
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
    }
	this->QoIsum[0] = 0;
	for (int i = this->chainIdx; i<this->maxChainLength; ++i){
        getline(indSamples,line);
        ss.str(line);
        getline(ss,temp,',');
        // ObsInd=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1Ind=std::stod(temp);
        getline(ss,temp,',');
        QoiIndt1=std::stod(temp);


        this->alpha = this->sampler->getAlpha(Likelihoodt0Ind, Likelihoodt1Ind);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0Ind=Likelihoodt1Ind;
            QoiIndt0=QoiIndt1;            
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
        this->QoIsum[0]+=QoiIndt0;
	}
	for (int i = 0; i < this->sampleSize; ++i){
		QoImean[i] = this->QoIsum[i]/(this->maxChainLength-this->numBurnin);
	}
    indSamples.close();
}


template <typename samplerType, typename solverType> 
class MLChain0i : public MCMCChain<samplerType, solverType> {
public:
	solverType* solverUpper;
	solverType* solverLower;

	MLChain0i(int maxChainLength_, int sampleSize_, solverType* solver_, solverType* solverUpper_, solverType* solverLower_, double beta_=1.0) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_, beta_), solverUpper(solverUpper_), solverLower(solverLower_){};
	~MLChain0i(){};

	virtual void updateQoI();
    // virtual void runStep(std::default_random_engine* generator);
    // virtual void run(double QoImean[], std::default_random_engine* generator);
    virtual void generateIndSamples(int targetSampleSize, int rank, int l, std::default_random_engine* generator);
    virtual void runInd(double QoImean[], int startLocation, int l, std::default_random_engine* generator);
};

template <typename samplerType, typename solverType> 
void MLChain0i<samplerType, solverType>::updateQoI(){
	for (int i = 0; i < this->sampleSize; ++i){
		solverUpper->samples[i] = this->solver->samples[i];
		solverLower->samples[i] = this->solver->samples[i];
	}
	solverUpper->solve(1);
	solverLower->solve(1);
	this->QoI[0] = solverUpper->solve4QoI()-solverLower->solve4QoI();
}

// template <typename samplerType, typename solverType> 
// void MLChain0i<samplerType, solverType>::runStep(std::default_random_engine* generator){
// 	this->chainIdx += 1;
// 	this->sampleProposal();
// 	this->solver->solve();
// 	this->updatelnLikelihood();
// 	this->checkAcceptance(generator);
// 	if (this->accepted == 1){
// 		this->lnLikelihoodt0 = this->lnLikelihoodt1;
// 		updateQoI();
// 	}
// 	if (this->chainIdx > this->numBurnin){
// 		this->QoIsum[0] += this->QoI[0];
//         std::cout << "MCMCChain " << this->QoI[0] << std::endl;
// 	}
// 	this->acceptanceRate = this->acceptedNum/this->chainIdx;
// }

// template <typename samplerType, typename solverType> 
// void MLChain0i<samplerType, solverType>::run(double QoImean[], std::default_random_engine* generator){
// 	this->chainInit(generator);
// 	this->QoIsum[0] = 0.0;
// 	for (int i = 1; i < this->maxChainLength+1; ++i){
// 		runStep(generator);
// 	}
// 	for (int i = 0; i < this->sampleSize; ++i){
// 		QoImean[i] = this->QoIsum[i]/(this->maxChainLength-this->numBurnin);
// 	}
// 	// std::cout << "acceptanceRate: " << acceptanceRate << std::endl;
// }



template <typename samplerType, typename solverType> 
void MLChain0i<samplerType, solverType>::generateIndSamples(int targetSampleSize, int rank, int l, std::default_random_engine* generator)
{
    std::string outputfile = "chain0";
    outputfile.append(std::to_string(l));
    outputfile.append("_ind_samples_");
    outputfile.append(std::to_string(rank));

    std::ofstream myfile;
    myfile.open(outputfile, std::ios_base::app);
    myfile<<"Obs,Likelihood,QoI,Sampels"<<std::endl;

    double Workspace;
	double initialSample[this->sampleSize];
	this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));
	this->solver->priorSample(initialSample);
	for (int i = 0; i < this->sampleSize; ++i){
		this->solver->samples[i] = initialSample[i];
		solverUpper->samples[i] = this->solver->samples[i];
		solverLower->samples[i] = this->solver->samples[i];
	}
	this->solver->solve();
	Workspace=this->solver->solve4Obs();myfile<<Workspace;
    Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
	solverUpper->solve(1);
	solverLower->solve(1);
	Workspace=solverUpper->solve4QoI()-solverLower->solve4QoI();myfile<<","<<Workspace;

    for (int i = 0; i<this->sampleSize; ++i){
        Workspace=this->solver->samples[i];myfile<<","<<Workspace;
    }
    myfile << std::endl;

	for (int i = 1; i < targetSampleSize; ++i){
        this->sampleProposal();
        for (int i = 0; i < this->sampleSize; ++i){
            solverUpper->samples[i] = this->solver->samples[i];
            solverLower->samples[i] = this->solver->samples[i];
        }
        this->solver->solve();
        Workspace=this->solver->solve4Obs();myfile<<Workspace;
        Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
        solverUpper->solve(1);
        solverLower->solve(1);
        Workspace=solverUpper->solve4QoI()-solverLower->solve4QoI();myfile<<","<<Workspace;

        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<","<<Workspace;
        }
        myfile << std::endl;
	}
    myfile.close();
} 

template <typename samplerType, typename solverType> 
void MLChain0i<samplerType, solverType>::runInd(double QoImean[], int startLocation, int l, std::default_random_engine* generator){
    std::string samplesCSV="chain0";
    samplesCSV.append(std::to_string(l));
    samplesCSV.append("_ind_samples");
    std::ifstream indSamples(samplesCSV);

    std::string line;
    std::string temp;
    getline(indSamples,line);

    //go to position
    for (int i=0; i<startLocation;++i)
    {
        getline(indSamples,line);
    }
    std::stringstream ss(line);
    getline(ss,temp,',');
    // double ObsInd=std::stod(temp);
    getline(ss,temp,',');
    double Likelihoodt0Ind=std::stod(temp);
    double Likelihoodt1Ind=Likelihoodt0Ind;
    getline(ss,temp,',');
    double QoiIndt0=std::stod(temp);
    double QoiIndt1=QoiIndt0;

	
    this->chainIdx=1;
    if (this->chainIdx < this->numBurnin)
    {
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);
        getline(ss,temp,',');
        // ObsInd=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1Ind=std::stod(temp);
        getline(ss,temp,',');
        QoiIndt1=std::stod(temp);


        this->alpha = this->sampler->getAlpha(Likelihoodt0Ind, Likelihoodt1Ind);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0Ind=Likelihoodt1Ind;
            QoiIndt0=QoiIndt1;
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
    }
	this->QoIsum[0] = 0;
	for (int i = this->chainIdx; i<this->maxChainLength; ++i){
        getline(indSamples,line);
        ss.str(line);
        getline(ss,temp,',');
        // ObsInd=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1Ind=std::stod(temp);
        getline(ss,temp,',');
        QoiIndt1=std::stod(temp);


        this->alpha = this->sampler->getAlpha(Likelihoodt0Ind, Likelihoodt1Ind);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0Ind=Likelihoodt1Ind;
            QoiIndt0=QoiIndt1;            
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
        this->QoIsum[0]+=QoiIndt0;
	}
	for (int i = 0; i < this->sampleSize; ++i){
		QoImean[i] = this->QoIsum[i]/(this->maxChainLength-this->numBurnin);
	}
    indSamples.close();
}

template <typename samplerType, typename solverType> 
class MLChaini0Upper : public MCMCChain<samplerType, solverType> {
public:
	double* A1sum;
	double* A3sum;
	double* A6sum;
	double* A7sum;
	double PhiUpper;
	double PhiLower;
	double I;

	solverType* samplerLower;
	solverType* solverUpper;

	MLChaini0Upper(int maxChainLength_, int sampleSize_, solverType* solver_, solverType* samplerLower_, solverType* solverUpper_, double beta_=1.0) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_, beta_), samplerLower(samplerLower_), solverUpper(solverUpper_){
		A1sum = new double[sampleSize_];
		A3sum = new double[sampleSize_];
		A6sum = new double[sampleSize_];
		A7sum = new double[sampleSize_];
	};
	~MLChaini0Upper(){
		delete[] A1sum;
		delete[] A3sum;
		delete[] A6sum;
		delete[] A7sum;
	};

	virtual void chainInit(std::default_random_engine* generator);
	virtual void updateQoI();
	virtual void runStep(std::default_random_engine* generator);
	virtual void run(double A1mean[], double A3mean[], double A6mean[], double A7mean[], std::default_random_engine* generator);
    virtual void generateIndSamples(int targetSampleSize, int rank, int l, std::default_random_engine* generator);
    virtual void runInd(double A1mean[], double A3mean[], double A6mean[], double A7mean[], int startLocation, int l, std::default_random_engine* generator);
};

template <typename samplerType, typename solverType> 
void MLChaini0Upper<samplerType, solverType>::chainInit(std::default_random_engine* generator){
	double initialSample[this->sampleSize];
	this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));
	this->solver->priorSample(initialSample);
	for (int i = 0; i < this->sampleSize; ++i){
		this->solver->samples[i] = initialSample[i];
	}
	this->solver->solve();
	this->solver->solve4Obs();
	this->lnLikelihoodt0 = this->solver->lnLikelihood();
	this->lnLikelihoodt1 = this->lnLikelihoodt0;
	updateQoI();
	A1sum[0] = 0;//I*(1-exp(this->PhiUpper-this->PhiLower))*this->QoI[0];
	A3sum[0] = 0;//I*(exp(this->PhiUpper-this->PhiLower)-1);
	A6sum[0] = 0;//I*exp(this->PhiUpper-this->PhiLower)*this->QoI[0];
	A7sum[0] = 0;//(1-I)*this->QoI[0];
	if (std::isnan(A1sum[0]) || std::isnan(A3sum[0]) || std::isnan(A6sum[0]) || std::isnan(A7sum[0])){
		std::cout << "MLChaini0Upper encountered nan" << std::endl;
		std::cin.get();
	}
}

template <typename samplerType, typename solverType> 
void MLChaini0Upper<samplerType, solverType>::updateQoI(){
	for (int i = 0; i < this->sampleSize; ++i){
		solverUpper->samples[i] = this->solver->samples[i];
		samplerLower->samples[i] = this->solver->samples[i];
	}
	solverUpper->solve(1);
	samplerLower->solve();
	this->PhiUpper = -this->lnLikelihoodt1;
	this->PhiLower = -samplerLower->lnLikelihood();
	if (this->PhiUpper - this->PhiLower <= 0){
		I = 1.0;
	} else {
		I = 0.0;
	}
	this->QoI[0] = solverUpper->solve4QoI();
}

template <typename samplerType, typename solverType> 
void MLChaini0Upper<samplerType, solverType>::runStep(std::default_random_engine* generator){
	this->chainIdx += 1;
	this->sampleProposal();
	this->solver->solve();
	this->updatelnLikelihood();
	this->checkAcceptance(generator);
	if (this->accepted == 1){
		this->lnLikelihoodt0 = this->lnLikelihoodt1;
		updateQoI();
	}
	if (this->chainIdx > this->numBurnin){
		A1sum[0] += I*(1-exp(std::min(this->PhiUpper-this->PhiLower, 0.0)))*this->QoI[0];
		A3sum[0] += I*(exp(std::min(this->PhiUpper-this->PhiLower, 0.0))-1.0);
		A6sum[0] += I*exp(std::min(this->PhiUpper-this->PhiLower, 0.0))*this->QoI[0];
		A7sum[0] += (1.0-I)*this->QoI[0];
	}
    // std::cout << "A1sum: " << A1sum[0] << std::endl;
    // std::cout << "i0upper chain " << this->PhiUpper << " " << this->PhiLower << " " << this->QoI[0] << " " << I*(1-exp(std::min(this->PhiUpper-this->PhiLower, 0.0)))*this->QoI[0] << std::endl;
	if (std::isnan(A1sum[0]) || std::isnan(A3sum[0]) || std::isnan(A6sum[0]) || std::isnan(A7sum[0])){
		std::cout << "MLChaini0Upper encountered nan" << std::endl;
		std::cin.get();
	}
	this->acceptanceRate = this->acceptedNum/this->chainIdx;
}

template <typename samplerType, typename solverType> 
void MLChaini0Upper<samplerType, solverType>::run(double A1mean[], double A3mean[], double A6mean[], double A7mean[], std::default_random_engine* generator){
	this->chainInit(generator);
	for (int i = 1; i < this->maxChainLength+1; ++i){
		runStep(generator);
	}
	for (int i = 0; i < this->sampleSize; ++i){
		A1mean[0] = A1sum[0] / (this->maxChainLength-this->numBurnin);
		A3mean[0] = A3sum[0] / (this->maxChainLength-this->numBurnin);
		A6mean[0] = A6sum[0] / (this->maxChainLength-this->numBurnin);
		A7mean[0] = A7sum[0] / (this->maxChainLength-this->numBurnin);
	}
	std::cout << "acceptanceRate: " << this->acceptanceRate << std::endl;
}


template <typename samplerType, typename solverType> 
void MLChaini0Upper<samplerType, solverType>::generateIndSamples(int targetSampleSize, int rank, int l, std::default_random_engine* generator)
{
    std::string outputfile = "chain";
    outputfile.append(std::to_string(l));
    outputfile.append("0Upper_ind_samples_");
    outputfile.append(std::to_string(rank));

    std::ofstream myfile;
    myfile.open(outputfile, std::ios_base::app);
    myfile<<"ObsUpper,ObsLower,LikelihoodUpper,LikelihoodLower,QoI,samples"<<std::endl;

    double Workspace;
	double initialSample[this->sampleSize];
	this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));
	this->solver->priorSample(initialSample);
	for (int i = 0; i < this->sampleSize; ++i){
		this->solver->samples[i] = initialSample[i];
		solverUpper->samples[i] = this->solver->samples[i];
		samplerLower->samples[i] = this->solver->samples[i];
	}

	this->solver->solve();
	samplerLower->solve();
	Workspace=this->solver->solve4Obs();myfile<<Workspace;
    Workspace=samplerLower->solve4Obs();myfile<<","<<Workspace;
    Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
    Workspace=samplerLower->lnLikelihood();myfile<<","<<Workspace;
	solverUpper->solve(1);
	Workspace=solverUpper->solve4QoI();myfile<<","<<Workspace;


    for (int i = 0; i<this->sampleSize; ++i){
        Workspace=this->solver->samples[i];myfile<<","<<Workspace;
    }
    myfile << std::endl;

	for (int i = 1; i < targetSampleSize; ++i){
        this->sampleProposal();
        for (int i = 0; i < this->sampleSize; ++i){
            solverUpper->samples[i] = this->solver->samples[i];
            samplerLower->samples[i] = this->solver->samples[i];
        }
        this->solver->solve();
        samplerLower->solve();
        Workspace=this->solver->solve4Obs();myfile<<Workspace;
        Workspace=samplerLower->solve4Obs();myfile<<","<<Workspace;
        Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
        Workspace=samplerLower->lnLikelihood();myfile<<","<<Workspace;
        solverUpper->solve(1);
        Workspace=solverUpper->solve4QoI();myfile<<","<<Workspace;


        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<","<<Workspace;
        }
        myfile << std::endl;
    }
    myfile.close();
} 

template <typename samplerType, typename solverType> 
void MLChaini0Upper<samplerType, solverType>::runInd(double A1mean[], double A3mean[], double A6mean[], double A7mean[], int startLocation, int l, std::default_random_engine* generator)
{
    std::string samplesCSV="chain";
    samplesCSV.append(std::to_string(l));
    samplesCSV.append("0Upper_ind_samples");
    std::ifstream indSamples(samplesCSV);

    std::string line;
    std::string temp;
    getline(indSamples,line);

    //go to position
    for (int i=0; i<startLocation;++i)
    {
        getline(indSamples,line);
    }
    std::stringstream ss(line);
    getline(ss,temp,',');
    // double ObsIndUpper=std::stod(temp);
    getline(ss,temp,',');
    // double ObsIndLower=std::stod(temp);
    getline(ss,temp,',');
    double Likelihoodt0IndUpper=std::stod(temp);
    double Likelihoodt1IndUpper=Likelihoodt0IndUpper;
    getline(ss,temp,',');
    double Likelihoodt0IndLower=std::stod(temp);
    double Likelihoodt1IndLower=Likelihoodt0IndLower;    
    getline(ss,temp,',');
    double QoiIndt0=std::stod(temp);
    double QoiIndt1=QoiIndt0;

    this->PhiLower=-Likelihoodt1IndLower;
    this->PhiUpper=-Likelihoodt1IndUpper;
    if (this->PhiUpper - this->PhiLower <= 0){
        I = 1.0;
    } else {
        I = 0.0;
    }	

    this->chainIdx=1;
    if (this->chainIdx < this->numBurnin)
    {
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);
        getline(ss,temp,',');
        // double ObsIndUpper=std::stod(temp);
        getline(ss,temp,',');
        // double ObsIndLower=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndUpper=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,',');
        QoiIndt1=std::stod(temp);


        this->alpha = this->sampler->getAlpha(Likelihoodt0IndUpper, Likelihoodt1IndUpper);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndUpper=Likelihoodt1IndUpper;
            QoiIndt0=QoiIndt1;
            this->PhiLower=-Likelihoodt1IndLower;
            this->PhiUpper=-Likelihoodt1IndUpper;
            if (this->PhiUpper - this->PhiLower <= 0){
                I = 1.0;
            } else {
                I = 0.0;
            }	
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
    }

    this->QoI[0] = QoiIndt0;
	A1sum[0] = 0;//I*(1-exp(this->PhiUpper-this->PhiLower))*this->QoI[0];
	A3sum[0] = 0;//I*(exp(this->PhiUpper-this->PhiLower)-1);
	A6sum[0] = 0;//I*exp(this->PhiUpper-this->PhiLower)*this->QoI[0];
	A7sum[0] = 0;//(1-I)*this->QoI[0];
	for (int i = this->chainIdx; i<this->maxChainLength; ++i){
        getline(indSamples,line);
        ss.str(line);
        getline(ss,temp,',');
        // double ObsIndUpper=std::stod(temp);
        getline(ss,temp,',');
        // double ObsIndLower=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndUpper=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,',');
        QoiIndt1=std::stod(temp);


        this->alpha = this->sampler->getAlpha(Likelihoodt0IndUpper, Likelihoodt1IndUpper);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndUpper=Likelihoodt1IndUpper;
            QoiIndt0=QoiIndt1;
            this->PhiLower=-Likelihoodt1IndLower;
            this->PhiUpper=-Likelihoodt1IndUpper;
            if (this->PhiUpper - this->PhiLower <= 0){
                I = 1.0;
            } else {
                I = 0.0;
            }
            this->QoI[0] = QoiIndt0;
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
        A1sum[0] += I*(1-exp(std::min(this->PhiUpper-this->PhiLower, 0.0)))*this->QoI[0];
        A3sum[0] += I*(exp(std::min(this->PhiUpper-this->PhiLower, 0.0))-1.0);
        A6sum[0] += I*exp(std::min(this->PhiUpper-this->PhiLower, 0.0))*this->QoI[0];
        A7sum[0] += (1.0-I)*this->QoI[0];
	}
	for (int i = 0; i < this->sampleSize; ++i){
		A1mean[0] = A1sum[0] / (this->maxChainLength-this->numBurnin);
		A3mean[0] = A3sum[0] / (this->maxChainLength-this->numBurnin);
		A6mean[0] = A6sum[0] / (this->maxChainLength-this->numBurnin);
		A7mean[0] = A7sum[0] / (this->maxChainLength-this->numBurnin);
	}
    indSamples.close();
}

template <typename samplerType, typename solverType> 
class MLChaini0Lower : public MCMCChain<samplerType, solverType> {
public:
	double* A2sum;
	double* A4sum;
	double* A8sum;
	double* A5sum;
	double PhiUpper;
	double PhiLower;
	double I;

	solverType* samplerUpper;
	solverType* solver0;

	MLChaini0Lower(int maxChainLength_, int sampleSize_, solverType* samplerUpper_, solverType* solver_, solverType* solver0_, double beta_=1.0) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_, beta_), samplerUpper(samplerUpper_), solver0(solver0_){
		A2sum = new double[sampleSize_];
		A4sum = new double[sampleSize_];
		A8sum = new double[sampleSize_];
		A5sum = new double[sampleSize_];
	};
	~MLChaini0Lower(){
		delete[] A2sum;
		delete[] A4sum;
		delete[] A8sum;
		delete[] A5sum;
	};

	virtual void chainInit(std::default_random_engine* generator);
	virtual void updateQoI();
	virtual void runStep(std::default_random_engine* generator);
	virtual void run(double A2mean[], double A4mean[], double A8mean[], double A5mean[], std::default_random_engine* generator);
    virtual void generateIndSamples(int targetSampleSize, int rank, int l, std::default_random_engine* generator);
    virtual void runInd(double A2mean[], double A4mean[], double A8mean[], double A5mean[], int startLocation, int l, std::default_random_engine* generator);
};

template <typename samplerType, typename solverType> 
void MLChaini0Lower<samplerType, solverType>::chainInit(std::default_random_engine* generator){
	double initialSample[this->sampleSize];
	this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));
	this->solver->priorSample(initialSample);
	for (int i = 0; i < this->sampleSize; ++i){
		this->solver->samples[i] = initialSample[i];
	}
	this->solver->solve();
	this->solver->solve4Obs();
	this->lnLikelihoodt0 = this->solver->lnLikelihood();
	this->lnLikelihoodt1 = this->lnLikelihoodt0;
	updateQoI();
	A2sum[0] = 0;//(1-I)*(exp(this->PhiLower-this->PhiUpper)-1)*this->QoI[0];
	A4sum[0] = 0;//I*this->QoI[0];
	A8sum[0] = 0;//(1-I)*exp(this->PhiLower-this->PhiUpper)*this->QoI[0];
	A5sum[0] = 0;//(1-I)*(1-exp(this->PhiLower-this->PhiUpper));
	if (std::isnan(A2sum[0]) || std::isnan(A4sum[0]) || std::isnan(A8sum[0]) || std::isnan(A5sum[0])){
		std::cout << "MLChaini0Lower encountered nan" << std::endl;
		std::cin.get();
	}
}

template <typename samplerType, typename solverType> 
void MLChaini0Lower<samplerType, solverType>::updateQoI(){
	for (int i = 0; i < this->sampleSize; ++i){
		solver0->samples[i] = this->solver->samples[i];
		samplerUpper->samples[i] = this->solver->samples[i];
	}
	solver0->solve();
	samplerUpper->solve();
	this->PhiLower = -this->lnLikelihoodt1;
	this->PhiUpper = -samplerUpper->lnLikelihood();
	if (this->PhiUpper - this->PhiLower <= 0){
		I = 1.0;
	} else {
		I = 0.0;
	}
	this->QoI[0] = solver0->solve4QoI();
}

template <typename samplerType, typename solverType> 
void MLChaini0Lower<samplerType, solverType>::runStep(std::default_random_engine* generator){
	this->chainIdx += 1;
	this->sampleProposal();
	this->solver->solve();
	this->updatelnLikelihood();
	this->checkAcceptance(generator);
	if (this->accepted == 1){
		this->lnLikelihoodt0 = this->lnLikelihoodt1;
		updateQoI();
	}
	if (this->chainIdx > this->numBurnin){
		A2sum[0] += (1-I)*(exp(std::min(this->PhiLower-this->PhiUpper, 0.0))-1)*this->QoI[0];
		A4sum[0] += I*this->QoI[0];
		A8sum[0] += (1-I)*exp(std::min(this->PhiLower-this->PhiUpper, 0.0))*this->QoI[0];
		A5sum[0] += (1-I)*(1-exp(std::min(this->PhiLower-this->PhiUpper, 0.0)));
    }
    std::cout << A2sum[0] << " " << (1-I)*(exp(std::min(this->PhiLower-this->PhiUpper, 0.0))-1) << " " << this->QoI[0] << std::endl;
	if (std::isnan(A2sum[0]) || std::isnan(A4sum[0]) || std::isnan(A8sum[0]) || std::isnan(A5sum[0])){
		std::cout << "MLChaini0Lower encountered nan" << std::endl;
		std::cin.get();
	}
	this->acceptanceRate = this->acceptedNum/this->chainIdx;
}

template <typename samplerType, typename solverType> 
void MLChaini0Lower<samplerType, solverType>::run(double A2mean[], double A4mean[], double A8mean[], double A5mean[], std::default_random_engine* generator){
	this->chainInit(generator);
	for (int i = 1; i < this->maxChainLength+1; ++i){
		runStep(generator);
	}
	for (int i = 0; i < this->sampleSize; ++i){
		A2mean[0] = A2sum[0] / (this->maxChainLength-this->numBurnin);
		A4mean[0] = A4sum[0] / (this->maxChainLength-this->numBurnin);
		A8mean[0] = A8sum[0] / (this->maxChainLength-this->numBurnin);
		A5mean[0] = A5sum[0] / (this->maxChainLength-this->numBurnin);
	}
	std::cout << " " << (this->maxChainLength-this->numBurnin) << "acceptanceRate: " << this->acceptanceRate << std::endl;
}


template <typename samplerType, typename solverType> 
void MLChaini0Lower<samplerType, solverType>::generateIndSamples(int targetSampleSize, int rank, int l, std::default_random_engine* generator)
{
    std::string outputfile = "chain";
    outputfile.append(std::to_string(l));
    outputfile.append("0Lower_ind_samples_");
    outputfile.append(std::to_string(rank));

    std::ofstream myfile;
    myfile.open(outputfile, std::ios_base::app);
    myfile<<"ObsUpper,ObsLower,LikelihoodUpper,LikelihoodLower,QoI,samples"<<std::endl;

    double Workspace;
	double initialSample[this->sampleSize];
	this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));
	this->solver->priorSample(initialSample);
	for (int i = 0; i < this->sampleSize; ++i){
		this->solver->samples[i] = initialSample[i];
		solver0->samples[i] = this->solver->samples[i];
		samplerUpper->samples[i] = this->solver->samples[i];
	}
	this->solver->solve();
	samplerUpper->solve();
    Workspace=samplerUpper->solve4Obs();myfile<<Workspace;
	Workspace=this->solver->solve4Obs();myfile<<","<<Workspace;
    Workspace=samplerUpper->lnLikelihood();myfile<<","<<Workspace;
    Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
	solver0->solve(1);
	Workspace=solver0->solve4QoI();myfile<<","<<Workspace;


    for (int i = 0; i<this->sampleSize; ++i){
        Workspace=this->solver->samples[i];myfile<<","<<Workspace;
    }
    myfile << std::endl;

	for (int i = 1; i < targetSampleSize; ++i){
        this->sampleProposal();
        for (int i = 0; i < this->sampleSize; ++i){
            solver0->samples[i] = this->solver->samples[i];
            samplerUpper->samples[i] = this->solver->samples[i];
        }
        this->solver->solve();
        samplerUpper->solve();
        Workspace=samplerUpper->solve4Obs();myfile<<Workspace;
        Workspace=this->solver->solve4Obs();myfile<<","<<Workspace;
        Workspace=samplerUpper->lnLikelihood();myfile<<","<<Workspace;
        Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
        solver0->solve(1);
        Workspace=solver0->solve4QoI();myfile<<","<<Workspace;


        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<","<<Workspace;
        }
        myfile << std::endl;
    }
    myfile.close();
} 


template <typename samplerType, typename solverType> 
void MLChaini0Lower<samplerType, solverType>::runInd(double A2mean[], double A4mean[], double A8mean[], double A5mean[], int startLocation, int l, std::default_random_engine* generator)
{
    std::string samplesCSV="chain";
    samplesCSV.append(std::to_string(l));
    samplesCSV.append("0Lower_ind_samples");
    std::ifstream indSamples(samplesCSV);

    std::string line;
    std::string temp;
    getline(indSamples,line);

    //go to position
    for (int i=0; i<startLocation;++i)
    {
        getline(indSamples,line);
    }
    std::stringstream ss(line);
    getline(ss,temp,',');
    // double ObsIndUpper=std::stod(temp);
    getline(ss,temp,',');
    // double ObsIndLower=std::stod(temp);
    getline(ss,temp,',');
    double Likelihoodt0IndUpper=std::stod(temp);
    double Likelihoodt1IndUpper=Likelihoodt0IndUpper;
    getline(ss,temp,',');
    double Likelihoodt0IndLower=std::stod(temp);
    double Likelihoodt1IndLower=Likelihoodt0IndLower;  
    getline(ss,temp,',');
    double QoiIndt0=std::stod(temp);
    double QoiIndt1=QoiIndt0;
  
    this->PhiLower=-Likelihoodt1IndLower;
    this->PhiUpper=-Likelihoodt1IndUpper;
    if (this->PhiUpper - this->PhiLower <= 0){
        I = 1.0;
    } else {
        I = 0.0;
    }	

    this->chainIdx=1;
    if (this->chainIdx < this->numBurnin)
    {
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);
        getline(ss,temp,',');
        // double ObsIndUpper=std::stod(temp);
        getline(ss,temp,',');
        // double ObsIndLower=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndUpper=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,',');
        QoiIndt1=std::stod(temp);


        this->alpha = this->sampler->getAlpha(Likelihoodt0IndLower, Likelihoodt1IndLower);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndLower=Likelihoodt1IndLower;
            QoiIndt0=QoiIndt1;
            this->PhiLower=-Likelihoodt1IndLower;
            this->PhiUpper=-Likelihoodt1IndUpper;
            if (this->PhiUpper - this->PhiLower <= 0){
                I = 1.0;
            } else {
                I = 0.0;
            }	
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
    }

    this->QoI[0] = QoiIndt0;
	A2sum[0] = 0;//I*(1-exp(this->PhiUpper-this->PhiLower))*this->QoI[0];
	A4sum[0] = 0;//I*(exp(this->PhiUpper-this->PhiLower)-1);
	A8sum[0] = 0;//I*exp(this->PhiUpper-this->PhiLower)*this->QoI[0];
	A5sum[0] = 0;//(1-I)*this->QoI[0];

	for (int i = this->chainIdx; i<this->maxChainLength; ++i){
        getline(indSamples,line);
        ss.str(line);
        getline(ss,temp,',');
        // double ObsIndUpper=std::stod(temp);
        getline(ss,temp,',');
        // double ObsIndLower=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndUpper=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,',');
        QoiIndt1=std::stod(temp);


        this->alpha = this->sampler->getAlpha(Likelihoodt0IndLower, Likelihoodt1IndLower);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndLower=Likelihoodt1IndLower;
            QoiIndt0=QoiIndt1;
            this->PhiLower=-Likelihoodt1IndLower;
            this->PhiUpper=-Likelihoodt1IndUpper;
            if (this->PhiUpper - this->PhiLower <= 0){
                I = 1.0;
            } else {
                I = 0.0;
            }
            this->QoI[0] = QoiIndt0;
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
		A2sum[0] += (1-I)*(exp(std::min(this->PhiLower-this->PhiUpper, 0.0))-1)*this->QoI[0];
		A4sum[0] += I*this->QoI[0];
		A8sum[0] += (1-I)*exp(std::min(this->PhiLower-this->PhiUpper, 0.0))*this->QoI[0];
		A5sum[0] += (1-I)*(1-exp(std::min(this->PhiLower-this->PhiUpper, 0.0)));
	}
	for (int i = 0; i < this->sampleSize; ++i){
		A2mean[0] = A2sum[0] / (this->maxChainLength-this->numBurnin);
		A4mean[0] = A4sum[0] / (this->maxChainLength-this->numBurnin);
		A8mean[0] = A8sum[0] / (this->maxChainLength-this->numBurnin);
		A5mean[0] = A5sum[0] / (this->maxChainLength-this->numBurnin);
	}
    indSamples.close();
}

template <typename samplerType, typename solverType> 
class MLChainijLower : public MCMCChain<samplerType, solverType> {
public:
	double* A2sum;
	double* A4sum;
	double* A8sum;
	double* A5sum;
	double PhiUpper;
	double PhiLower;
	double I;

	solverType* samplerUpper;
	solverType* solverUpper;
	solverType* solverLower;

	MLChainijLower(int maxChainLength_, int sampleSize_, solverType* samplerUpper_, solverType* solver_, solverType* solverUpper_, solverType* solverLower_, double beta_=1.0) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_, beta_), samplerUpper(samplerUpper_), solverUpper(solverUpper_), solverLower(solverLower_){
		A2sum = new double[sampleSize_];
		A4sum = new double[sampleSize_];
		A8sum = new double[sampleSize_];
		A5sum = new double[sampleSize_];
	};
	~MLChainijLower(){
		delete[] A2sum;
		delete[] A4sum;
		delete[] A8sum;
		delete[] A5sum;
	};

	virtual void chainInit(std::default_random_engine* generator);
	virtual void updateQoI();
	virtual void runStep(std::default_random_engine* generator);
	virtual void run(double A2mean[], double A4mean[], double A8mean[], double A5mean[], std::default_random_engine* generator);
    virtual void generateIndSamples(int targetSampleSize, int rank, int l1, int l2, std::default_random_engine* generator);
	virtual void runInd(double A2mean[], double A4mean[], double A8mean[], double A5mean[], int startLocation, int l1, int l2, std::default_random_engine* generator);
};

template <typename samplerType, typename solverType> 
void MLChainijLower<samplerType, solverType>::chainInit(std::default_random_engine* generator){
	double initialSample[this->sampleSize];
	this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));
	this->solver->priorSample(initialSample);
	for (int i = 0; i < this->sampleSize; ++i){
		this->solver->samples[i] = initialSample[i];
	}
	this->solver->solve();
	this->solver->solve4Obs();
	this->lnLikelihoodt0 = this->solver->lnLikelihood();
	this->lnLikelihoodt1 = this->lnLikelihoodt0;
	updateQoI();
	A2sum[0] = 0; //(1-I)*(exp(this->PhiLower-this->PhiUpper)-1)*this->QoI[0];
	A4sum[0] = 0;//I*this->QoI[0];
	A8sum[0] = 0;//(1-I)*exp(this->PhiLower-this->PhiUpper)*this->QoI[0];
	A5sum[0] = 0;//(1-I)*(1-exp(this->PhiLower-this->PhiUpper));
	if (std::isnan(A2sum[0]) || std::isnan(A4sum[0]) || std::isnan(A8sum[0]) || std::isnan(A5sum[0])){
		std::cout << "MLChainijLower encountered nan" << std::endl;
		std::cin.get();
	}
}

template <typename samplerType, typename solverType> 
void MLChainijLower<samplerType, solverType>::updateQoI(){
	for (int i = 0; i < this->sampleSize; ++i){
		solverUpper->samples[i] = this->solver->samples[i];
		solverLower->samples[i] = this->solver->samples[i];
		samplerUpper->samples[i] = this->solver->samples[i];
	}
	solverUpper->solve(1);
	solverLower->solve(1);
	samplerUpper->solve();
	this->PhiLower = -this->lnLikelihoodt1;
	this->PhiUpper = -samplerUpper->lnLikelihood();
	if (this->PhiUpper - this->PhiLower <= 0){
		I = 1.0;
	} else {
		I = 0.0;
	}
	this->QoI[0] = solverUpper->solve4QoI()-solverLower->solve4QoI();
}

template <typename samplerType, typename solverType> 
void MLChainijLower<samplerType, solverType>::runStep(std::default_random_engine* generator){
	this->chainIdx += 1;
	this->sampleProposal();
	this->solver->solve();
	this->updatelnLikelihood();
	this->checkAcceptance(generator);
	if (this->accepted == 1){
		this->lnLikelihoodt0 = this->lnLikelihoodt1;
		updateQoI();
	} 
	if (this->chainIdx > this->numBurnin){
		A2sum[0] += (1-I)*(exp(std::min(this->PhiLower-this->PhiUpper, 0.0))-1)*this->QoI[0];
		A4sum[0] += I*this->QoI[0];
		A8sum[0] += (1-I)*exp(std::min(this->PhiLower-this->PhiUpper, 0.0))*this->QoI[0];
		A5sum[0] += (1-I)*(1-exp(std::min(this->PhiLower-this->PhiUpper, 0.0)));
	}
    std::cout << A2sum[0] << " " << (1-I)*(exp(std::min(this->PhiLower-this->PhiUpper, 0.0))-1) << " " << this->QoI[0] << std::endl;
	if (std::isnan(A2sum[0]) || std::isnan(A4sum[0]) || std::isnan(A8sum[0]) || std::isnan(A5sum[0])){
		std::cout << "MLChainijLower encountered nan" << std::endl;
		std::cin.get();
	}
	this->acceptanceRate = this->acceptedNum/this->chainIdx;
}

template <typename samplerType, typename solverType> 
void MLChainijLower<samplerType, solverType>::run(double A2mean[], double A4mean[], double A8mean[], double A5mean[], std::default_random_engine* generator){
	this->chainInit(generator);
	for (int i = 1; i < this->maxChainLength+1; ++i){
		runStep(generator);
	}
	for (int i = 0; i < this->sampleSize; ++i){
		A2mean[0] = A2sum[0] / (this->maxChainLength-this->numBurnin);
		A4mean[0] = A4sum[0] / (this->maxChainLength-this->numBurnin);
		A8mean[0] = A8sum[0] / (this->maxChainLength-this->numBurnin);
		A5mean[0] = A5sum[0] / (this->maxChainLength-this->numBurnin);
	}
	std::cout << "acceptanceRate: " << this->acceptanceRate << std::endl;
}

template <typename samplerType, typename solverType> 
void MLChainijLower<samplerType, solverType>::generateIndSamples(int targetSampleSize, int rank, int l1, int l2, std::default_random_engine* generator)
{
    std::string outputfile = "chain";
    outputfile.append(std::to_string(l1));
    outputfile.append(std::to_string(l2));
    outputfile.append("Lower_ind_samples_");
    outputfile.append(std::to_string(rank));

    std::ofstream myfile;
    myfile.open(outputfile, std::ios_base::app);
    myfile<<"ObsUpper,ObsLower,LikelihoodUpper,LikelihoodLower,QoI,samples"<<std::endl;

    double Workspace;
	double initialSample[this->sampleSize];
	this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));
	this->solver->priorSample(initialSample);
	for (int i = 0; i < this->sampleSize; ++i){
		this->solver->samples[i] = initialSample[i];
		solverUpper->samples[i] = this->solver->samples[i];
        solverLower->samples[i] = this->solver->samples[i];
		samplerUpper->samples[i] = this->solver->samples[i];
	}
	this->solver->solve();
	samplerUpper->solve();
    Workspace=samplerUpper->solve4Obs();myfile<<Workspace;
	Workspace=this->solver->solve4Obs();myfile<<","<<Workspace;
    Workspace=samplerUpper->lnLikelihood();myfile<<","<<Workspace;
    Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
	solverUpper->solve(1);
    solverLower->solve(1);
	Workspace=solverUpper->solve4QoI()-solverLower->solve4QoI();myfile<<","<<Workspace;


    for (int i = 0; i<this->sampleSize; ++i){
        Workspace=this->solver->samples[i];myfile<<","<<Workspace;
    }
    myfile << std::endl;

	for (int i = 1; i < targetSampleSize; ++i){
        this->sampleProposal();
        for (int i = 0; i < this->sampleSize; ++i){
            solverUpper->samples[i] = this->solver->samples[i];
            solverLower->samples[i] = this->solver->samples[i];
    		samplerUpper->samples[i] = this->solver->samples[i];
        }
        this->solver->solve();
        samplerUpper->solve();
        Workspace=samplerUpper->solve4Obs();myfile<<Workspace;
        Workspace=this->solver->solve4Obs();myfile<<","<<Workspace;
        Workspace=samplerUpper->lnLikelihood();myfile<<","<<Workspace;
        Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
        solverUpper->solve(1);
        solverLower->solve(1);
        Workspace=solverUpper->solve4QoI()-solverLower->solve4QoI();myfile<<","<<Workspace;


        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<","<<Workspace;
        }
        myfile << std::endl;
    }
    myfile.close();
} 

template <typename samplerType, typename solverType> 
void MLChainijLower<samplerType, solverType>::runInd(double A2mean[], double A4mean[], double A8mean[], double A5mean[], int startLocation, int l1, int l2, std::default_random_engine* generator)
{
    std::string samplesCSV="chain";
    samplesCSV.append(std::to_string(l1));
    samplesCSV.append(std::to_string(l2));
    samplesCSV.append("Lower_ind_samples");
    std::ifstream indSamples(samplesCSV);

    std::string line;
    std::string temp;
    getline(indSamples,line);

    //go to position
    for (int i=0; i<startLocation;++i)
    {
        getline(indSamples,line);
    }
    std::stringstream ss(line);
    getline(ss,temp,',');
    // double ObsIndUpper=std::stod(temp);
    getline(ss,temp,',');
    // double ObsIndLower=std::stod(temp);
    getline(ss,temp,',');
    double Likelihoodt0IndUpper=std::stod(temp);
    double Likelihoodt1IndUpper=Likelihoodt0IndUpper;
    getline(ss,temp,',');
    double Likelihoodt0IndLower=std::stod(temp);
    double Likelihoodt1IndLower=Likelihoodt0IndLower;    
    getline(ss,temp,',');
    double QoiIndt0=std::stod(temp);
    double QoiIndt1=QoiIndt0;

    this->PhiLower=-Likelihoodt1IndLower;
    this->PhiUpper=-Likelihoodt1IndUpper;
    if (this->PhiUpper - this->PhiLower <= 0){
        I = 1.0;
    } else {
        I = 0.0;
    }	

    this->chainIdx=1;
    if (this->chainIdx < this->numBurnin)
    {
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);
        getline(ss,temp,',');
        // double ObsIndUpper=std::stod(temp);
        getline(ss,temp,',');
        // double ObsIndLower=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndUpper=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,',');
        QoiIndt1=std::stod(temp);


        this->alpha = this->sampler->getAlpha(Likelihoodt0IndLower, Likelihoodt1IndLower);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndLower=Likelihoodt1IndLower;
            QoiIndt0=QoiIndt1;
            this->PhiLower=-Likelihoodt1IndLower;
            this->PhiUpper=-Likelihoodt1IndUpper;
            if (this->PhiUpper - this->PhiLower <= 0){
                I = 1.0;
            } else {
                I = 0.0;
            }	
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
    }

    this->QoI[0] = QoiIndt0;
	A2sum[0] = 0;//I*(1-exp(this->PhiUpper-this->PhiLower))*this->QoI[0];
	A4sum[0] = 0;//I*(exp(this->PhiUpper-this->PhiLower)-1);
	A8sum[0] = 0;//I*exp(this->PhiUpper-this->PhiLower)*this->QoI[0];
	A5sum[0] = 0;//(1-I)*this->QoI[0];

	for (int i = this->chainIdx; i<this->maxChainLength; ++i){
        getline(indSamples,line);
        ss.str(line);
        getline(ss,temp,',');
        // double ObsIndUpper=std::stod(temp);
        getline(ss,temp,',');
        // double ObsIndLower=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndUpper=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,',');
        QoiIndt1=std::stod(temp);

        this->alpha = this->sampler->getAlpha(Likelihoodt0IndLower, Likelihoodt1IndLower);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndLower=Likelihoodt1IndLower;
            QoiIndt0=QoiIndt1;
            this->PhiLower=-Likelihoodt1IndLower;
            this->PhiUpper=-Likelihoodt1IndUpper;
            if (this->PhiUpper - this->PhiLower <= 0){
                I = 1.0;
            } else {
                I = 0.0;
            }
            this->QoI[0] = QoiIndt0;
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
		A2sum[0] += (1-I)*(exp(std::min(this->PhiLower-this->PhiUpper, 0.0))-1)*this->QoI[0];
		A4sum[0] += I*this->QoI[0];
		A8sum[0] += (1-I)*exp(std::min(this->PhiLower-this->PhiUpper, 0.0))*this->QoI[0];
		A5sum[0] += (1-I)*(1-exp(std::min(this->PhiLower-this->PhiUpper, 0.0)));
	}
	for (int i = 0; i < this->sampleSize; ++i){
		A2mean[0] = A2sum[0] / (this->maxChainLength-this->numBurnin);
		A4mean[0] = A4sum[0] / (this->maxChainLength-this->numBurnin);
		A8mean[0] = A8sum[0] / (this->maxChainLength-this->numBurnin);
		A5mean[0] = A5sum[0] / (this->maxChainLength-this->numBurnin);
	}
    indSamples.close();
}

template <typename samplerType, typename solverType> 
class MLChainijUpper : public MCMCChain<samplerType, solverType> {
public:
	double* A1sum;
	double* A3sum;
	double* A6sum;
	double* A7sum;
	double PhiUpper;
	double PhiLower;
	double I;

	solverType* samplerLower;
	solverType* solverUpper;
	solverType* solverLower;

	MLChainijUpper(int maxChainLength_, int sampleSize_, solverType* solver_, solverType* samplerLower_, solverType* solverUpper_, solverType* solverLower_, double beta_=1.0) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_, beta_), samplerLower(samplerLower_), solverUpper(solverUpper_), solverLower(solverLower_){
		A1sum = new double[sampleSize_];
		A3sum = new double[sampleSize_];
		A6sum = new double[sampleSize_];
		A7sum = new double[sampleSize_];
	};
	~MLChainijUpper(){
		delete[] A1sum;
		delete[] A3sum;
		delete[] A6sum;
		delete[] A7sum;
	};

	virtual void chainInit(std::default_random_engine* generator);
	virtual void updateQoI();
	virtual void runStep(std::default_random_engine* generator);
	virtual void run(double A1mean[], double A3mean[], double A6mean[], double A7mean[], std::default_random_engine* generator);
    virtual void generateIndSamples(int targetSampleSize, int rank, int l1, int l2, std::default_random_engine* generator);
	virtual void runInd(double A1mean[], double A3mean[], double A6mean[], double A7mean[], int startLocation, int l1, int l2, std::default_random_engine* generator);
};

template <typename samplerType, typename solverType> 
void MLChainijUpper<samplerType, solverType>::chainInit(std::default_random_engine* generator){
	double initialSample[this->sampleSize];
	this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));
	this->solver->priorSample(initialSample);
	for (int i = 0; i < this->sampleSize; ++i){
		this->solver->samples[i] = initialSample[i];
	}
	this->solver->solve();
	this->solver->solve4Obs();
	this->lnLikelihoodt0 = this->solver->lnLikelihood();
	this->lnLikelihoodt1 = this->lnLikelihoodt0;
	updateQoI();
	A1sum[0] = 0; //I*(1-exp(this->PhiUpper-this->PhiLower))*this->QoI[0];
	A3sum[0] = 0; //I*(exp(this->PhiUpper-this->PhiLower)-1);
	A6sum[0] = 0; //I*exp(this->PhiUpper-this->PhiLower)*this->QoI[0];
	A7sum[0] = 0; //(1-I)*this->QoI[0];
	if (std::isnan(A1sum[0]) || std::isnan(A3sum[0]) || std::isnan(A6sum[0]) || std::isnan(A7sum[0])){
		std::cout << "MLChainijUpper encountered nan" << std::endl;
		std::cin.get();
	}
}

template <typename samplerType, typename solverType> 
void MLChainijUpper<samplerType, solverType>::updateQoI(){
	for (int i = 0; i < this->sampleSize; ++i){
		solverUpper->samples[i] = this->solver->samples[i];
		solverLower->samples[i] = this->solver->samples[i];
		samplerLower->samples[i] = this->solver->samples[i];
	}
	solverUpper->solve(1);
	solverLower->solve(1);
	samplerLower->solve();
	this->PhiUpper = -this->lnLikelihoodt1;
	this->PhiLower = -samplerLower->lnLikelihood();
	if (this->PhiUpper - this->PhiLower <= 0){
		I = 1.0;
	} else {
		I = 0.0;
	}
	this->QoI[0] = solverUpper->solve4QoI()-solverLower->solve4QoI();
}

template <typename samplerType, typename solverType> 
void MLChainijUpper<samplerType, solverType>::runStep(std::default_random_engine* generator){
	this->chainIdx += 1;
	this->sampleProposal();
	this->solver->solve();
	this->updatelnLikelihood();
	this->checkAcceptance(generator);
	if (this->accepted == 1){
		this->lnLikelihoodt0 = this->lnLikelihoodt1;
		updateQoI();
	} 
	if (this->chainIdx > this->numBurnin){
		A1sum[0] += I*(1-exp(std::min(this->PhiUpper-this->PhiLower, 0.0)))*this->QoI[0];
		A3sum[0] += I*(exp(std::min(this->PhiUpper-this->PhiLower, 0.0))-1);
		A6sum[0] += I*exp(std::min(this->PhiUpper-this->PhiLower, 0.0))*this->QoI[0];
		A7sum[0] += (1-I)*this->QoI[0];
	}
    std::cout << A1sum[0] << " " << I*(1-exp(std::min(this->PhiUpper-this->PhiLower, 0.0)))<< " " << this->QoI[0] << std::endl;
	if (std::isnan(A1sum[0]) || std::isnan(A3sum[0]) || std::isnan(A6sum[0]) || std::isnan(A7sum[0])){
		std::cout << "MLChainijUpper encountered nan" << std::endl;
		std::cin.get();
	}
	this->acceptanceRate = this->acceptedNum/this->chainIdx;
}

template <typename samplerType, typename solverType> 
void MLChainijUpper<samplerType, solverType>::run(double A1mean[], double A3mean[], double A6mean[], double A7mean[], std::default_random_engine* generator){
	this->chainInit(generator);
	for (int i = 1; i < this->maxChainLength+1; ++i){
		runStep(generator);
	}
	for (int i = 0; i < this->sampleSize; ++i){
		A1mean[0] = A1sum[0] / (this->maxChainLength-this->numBurnin);
		A3mean[0] = A3sum[0] / (this->maxChainLength-this->numBurnin);
		A6mean[0] = A6sum[0] / (this->maxChainLength-this->numBurnin);
		A7mean[0] = A7sum[0] / (this->maxChainLength-this->numBurnin);
	}
	std::cout << "acceptanceRate: " << this->acceptanceRate << std::endl;
}

template <typename samplerType, typename solverType> 
void MLChainijUpper<samplerType, solverType>::generateIndSamples(int targetSampleSize, int rank, int l1, int l2, std::default_random_engine* generator)
{
    std::string outputfile = "chain";
    outputfile.append(std::to_string(l1));
    outputfile.append(std::to_string(l2));
    outputfile.append("Upper_ind_samples_");
    outputfile.append(std::to_string(rank));

    std::ofstream myfile;
    myfile.open(outputfile, std::ios_base::app);
    myfile<<"ObsUpper,ObsLower,LikelihoodUpper,LikelihoodLower,QoI,samples"<<std::endl;

    double Workspace;
	double initialSample[this->sampleSize];
	this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));
	this->solver->priorSample(initialSample);
	for (int i = 0; i < this->sampleSize; ++i){
		this->solver->samples[i] = initialSample[i];
		solverUpper->samples[i] = this->solver->samples[i];
        solverLower->samples[i] = this->solver->samples[i];
		samplerLower->samples[i] = this->solver->samples[i];
	}
	this->solver->solve();
	samplerLower->solve();
    Workspace=this->solver->solve4Obs();myfile<<Workspace;
	Workspace=samplerLower->solve4Obs();myfile<<","<<Workspace;
    Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
    Workspace=samplerLower->lnLikelihood();myfile<<","<<Workspace;
	solverUpper->solve(1);
    solverLower->solve(1);
	Workspace=solverUpper->solve4QoI()-solverLower->solve4QoI();myfile<<","<<Workspace;


    for (int i = 0; i<this->sampleSize; ++i){
        Workspace=this->solver->samples[i];myfile<<","<<Workspace;
    }
    myfile << std::endl;

	for (int i = 1; i < targetSampleSize; ++i){
        this->sampleProposal();
        for (int i = 0; i < this->sampleSize; ++i){
            solverUpper->samples[i] = this->solver->samples[i];
            solverLower->samples[i] = this->solver->samples[i];
    		samplerLower->samples[i] = this->solver->samples[i];
        }
        this->solver->solve();
        samplerLower->solve();
        Workspace=this->solver->solve4Obs();myfile<<Workspace;
        Workspace=samplerLower->solve4Obs();myfile<<","<<Workspace;
        Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
        Workspace=samplerLower->lnLikelihood();myfile<<","<<Workspace;
        solverUpper->solve(1);
        solverLower->solve(1);
        Workspace=solverUpper->solve4QoI()-solverLower->solve4QoI();myfile<<","<<Workspace;


        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<","<<Workspace;
        }
        myfile << std::endl;
    }
    myfile.close();
} 


template <typename samplerType, typename solverType> 
void MLChainijUpper<samplerType, solverType>::runInd(double A1mean[], double A3mean[], double A6mean[], double A7mean[], int startLocation, int l1, int l2, std::default_random_engine* generator)
{
    std::string samplesCSV="chain";
    samplesCSV.append(std::to_string(l1));
    samplesCSV.append(std::to_string(l2));
    samplesCSV.append("Upper_ind_samples");
    std::ifstream indSamples(samplesCSV);

    std::string line;
    std::string temp;
    getline(indSamples,line);

    //go to position
    for (int i=0; i<startLocation;++i)
    {
        getline(indSamples,line);
    }
    std::stringstream ss(line);
    getline(ss,temp,',');
    // double ObsIndUpper=std::stod(temp);
    getline(ss,temp,',');
    // double ObsIndLower=std::stod(temp);
    getline(ss,temp,',');
    double Likelihoodt0IndUpper=std::stod(temp);
    double Likelihoodt1IndUpper=Likelihoodt0IndUpper;
    getline(ss,temp,',');
    double Likelihoodt0IndLower=std::stod(temp);
    double Likelihoodt1IndLower=Likelihoodt0IndLower;  
    getline(ss,temp,',');
    double QoiIndt0=std::stod(temp);
    double QoiIndt1=QoiIndt0;
  
    this->PhiLower=-Likelihoodt1IndLower;
    this->PhiUpper=-Likelihoodt1IndUpper;
    if (this->PhiUpper - this->PhiLower <= 0){
        I = 1.0;
    } else {
        I = 0.0;
    }		

    this->chainIdx=1;
    if (this->chainIdx < this->numBurnin)
    {
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);
        getline(ss,temp,',');
        // double ObsIndUpper=std::stod(temp);
        getline(ss,temp,',');
        // double ObsIndLower=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndUpper=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,',');
        QoiIndt1=std::stod(temp);


        this->alpha = this->sampler->getAlpha(Likelihoodt0IndUpper, Likelihoodt1IndUpper);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndUpper=Likelihoodt1IndUpper;
            QoiIndt0=QoiIndt1;
            this->PhiLower=-Likelihoodt1IndLower;
            this->PhiUpper=-Likelihoodt1IndUpper;
            if (this->PhiUpper - this->PhiLower <= 0){
                I = 1.0;
            } else {
                I = 0.0;
            }	
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
    }

    this->QoI[0] = QoiIndt0;
	A1sum[0] = 0;//I*(1-exp(this->PhiUpper-this->PhiLower))*this->QoI[0];
	A3sum[0] = 0;//I*(exp(this->PhiUpper-this->PhiLower)-1);
	A6sum[0] = 0;//I*exp(this->PhiUpper-this->PhiLower)*this->QoI[0];
	A7sum[0] = 0;//(1-I)*this->QoI[0];

	for (int i = this->chainIdx; i<this->maxChainLength; ++i){
        getline(indSamples,line);
        ss.str(line);
        getline(ss,temp,',');
        // double ObsIndUpper=std::stod(temp);
        getline(ss,temp,',');
        // double ObsIndLower=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndUpper=std::stod(temp);
        getline(ss,temp,',');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,',');
        QoiIndt1=std::stod(temp);


        this->alpha = this->sampler->getAlpha(Likelihoodt0IndUpper, Likelihoodt1IndUpper);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndUpper=Likelihoodt1IndUpper;
            QoiIndt0=QoiIndt1;
            this->PhiLower=-Likelihoodt1IndLower;
            this->PhiUpper=-Likelihoodt1IndUpper;
            if (this->PhiUpper - this->PhiLower <= 0){
                I = 1.0;
            } else {
                I = 0.0;
            }
            this->QoI[0] = QoiIndt0;
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
        A1sum[0] += I*(1-exp(std::min(this->PhiUpper-this->PhiLower, 0.0)))*this->QoI[0];
        A3sum[0] += I*(exp(std::min(this->PhiUpper-this->PhiLower, 0.0))-1.0);
        A6sum[0] += I*exp(std::min(this->PhiUpper-this->PhiLower, 0.0))*this->QoI[0];
        A7sum[0] += (1.0-I)*this->QoI[0];
	}
	for (int i = 0; i < this->sampleSize; ++i){
		A1mean[0] = A1sum[0] / (this->maxChainLength-this->numBurnin);
		A3mean[0] = A3sum[0] / (this->maxChainLength-this->numBurnin);
		A6mean[0] = A6sum[0] / (this->maxChainLength-this->numBurnin);
		A7mean[0] = A7sum[0] / (this->maxChainLength-this->numBurnin);
	}
    indSamples.close();
}