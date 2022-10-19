#pragma once

#include "MCMCChain.h"
#include <iostream>
#include <fstream>
#include <sstream>

template <typename samplerType, typename solverType> 
class MLChain_Uni_00 : public MCMCChain<samplerType, solverType> {
public:
	MLChain_Uni_00(int maxChainLength_, int sampleSize_, solverType* solver_) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_){};
	~MLChain_Uni_00(){};

    virtual void generateIndSamples(int targetSampleSize, int rank, std::default_random_engine* generator);
    virtual void runInd(double QoImean[], int startLocation, int rank, std::default_random_engine* generator);
};


template <typename samplerType, typename solverType> 
void MLChain_Uni_00<samplerType, solverType>::generateIndSamples(int targetSampleSize, int rank, std::default_random_engine* generator)
{
    std::string outputfile = "Uni_chain00_ind_samples_";
    outputfile.append(std::to_string(rank));
    std::ifstream cFile(outputfile);
    std::ofstream myfile;

    int readCount=0;
    bool completed=false;
    double Workspace;
    double Workspace_Obs[20];
    int Workspace_Obs_Size=10;
    double initialSample[this->sampleSize];
    this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));

    if (cFile.is_open()){
        std::string line;
        std::string line_back;
        getline(cFile, line);
        line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
        while(getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if (line[0] == '#' || line.empty()){
                continue; 
            }
            if (line[0] == 'c'){
                completed=true;
                std::cout << outputfile << " is completed" << std::endl;
                break;
            }
            readCount++;
            line_back=line;
        }
        if (completed != true){
            auto delimiterPos = line_back.find_last_of(";");
            auto lastSample = line_back.substr(delimiterPos+1);
            std::cout << "last sample: " << lastSample << std::endl << std::flush;
            this->solver->samples[0] = std::stod(lastSample);
            std::cout << "continue from last sample" << this->solver->samples[0] << std::endl << std::flush;
        }
        cFile.close();
        myfile.open(outputfile, std::ios_base::app);
    } else {
        cFile.close();
        myfile.open(outputfile, std::ios_base::app);

        myfile<<"Obs,Likelihood,QoI,samples"<<std::endl;
        this->solver->priorSample(initialSample);
        for (int i = 0; i < this->sampleSize; ++i){
            this->solver->samples[i] = initialSample[i];
        }
        this->solver->solve();
        Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_Size);
        myfile << Workspace_Obs[0];
        for (int i=1; i<20; i++){
            myfile<<"," << Workspace_Obs[i];
        }
        Workspace=this->solver->lnLikelihood();myfile<<";"<<Workspace;
        Workspace=this->solver->solve4QoI();myfile<<";"<<Workspace;

        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<";"<<Workspace;
        }
        myfile << std::endl;
    };

    if (completed != true){
        targetSampleSize = targetSampleSize-readCount;
        this->solver->priorSample(initialSample);
        for (int i = 0; i < this->sampleSize; ++i){
            this->solver->samples[i] = initialSample[i];
        }
        this->solver->solve();
        Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_Size);
        myfile << Workspace_Obs[0];
        for (int i=1; i<20; i++){
            myfile<<"," << Workspace_Obs[i];
        }
        Workspace=this->solver->lnLikelihood();myfile<<";"<<Workspace;
        Workspace=this->solver->solve4QoI();myfile<<";"<<Workspace;

        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<";"<<Workspace;
        }
        myfile << std::endl;

        for (int i = 1; i < targetSampleSize; ++i){
            this->sampleProposal();
            this->solver->solve();
            Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_Size);
            myfile << Workspace_Obs[0];
            for (int i=1; i<20; i++){
                myfile<<"," << Workspace_Obs[i];
            }
            Workspace=this->solver->lnLikelihood();myfile<<";"<<Workspace;
            Workspace=this->solver->solve4QoI();myfile<<";"<<Workspace;

            for (int i = 0; i<this->sampleSize; ++i){
                Workspace=this->solver->samples[i];myfile<<";"<<Workspace;
            }
            myfile << std::endl;
        }
        myfile << "complete";
    }
    myfile.close();
} 


template <typename samplerType, typename solverType> 
void MLChain_Uni_00<samplerType, solverType>::runInd(double QoImean[], int startLocation, int rank, std::default_random_engine* generator){
    double ObsInd[20];
    std::string inputfile = "Uni_chain00_ind_samples_";
    inputfile.append(std::to_string(rank));
    std::ifstream indSamples(inputfile);
    std::string line;
    std::string temp;
    getline(indSamples,line);

    //go to position
    for (int i=0; i<startLocation;++i)
    {
        getline(indSamples,line);
    }
    std::stringstream ss(line);
    for (int i=0; i<19; ++i){
        getline(ss,temp,',');
        ObsInd[i]=std::stod(temp);
    }
    getline(ss, temp, ';');
    ObsInd[19]=std::stod(temp);
    getline(ss,temp,';');
    double Likelihoodt0Ind=std::stod(temp);
    double Likelihoodt1Ind=Likelihoodt0Ind;
    getline(ss,temp,';');
    double QoiIndt0=std::stod(temp);
    double QoiIndt1=QoiIndt0;


    this->chainIdx=1;	
    for (int i=this->chainIdx; i < this->numBurnin; ++i)
    {
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);
        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsInd[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsInd[19]=std::stod(temp);

        getline(ss,temp,';');
        Likelihoodt1Ind=std::stod(temp);
        getline(ss,temp,';');
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
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);

        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsInd[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsInd[19]=std::stod(temp);

        getline(ss,temp,';');
        Likelihoodt1Ind=std::stod(temp);
        getline(ss,temp,';');
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
    std::cout << "L00 completed" << std::endl << std::flush;
}

template <typename samplerType, typename solverType> 
class MLChain_Uni_0i : public MCMCChain<samplerType, solverType> {
public:
    solverType* solverUpper;
    solverType* solverLower;
    
    MLChain_Uni_0i(int maxChainLength_, int sampleSize_, solverType* solver_, solverType* solverUpper_, solverType* solverLower_) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_), solverUpper(solverUpper_), solverLower(solverLower_){};
    ~MLChain_Uni_0i(){};
    
    virtual void updateQoI();
    // virtual void runStep(std::default_random_engine* generator);
    // virtual void run(double QoImean[], std::default_random_engine* generator);
    virtual void generateIndSamples(int targetSampleSize, int rank, int l, std::default_random_engine* generator);
    virtual void runInd(double QoImean[], int startLocation, int rank, int l, std::default_random_engine* generator);
};

template <typename samplerType, typename solverType> 
void MLChain_Uni_0i<samplerType, solverType>::updateQoI(){
	for (int i = 0; i < this->sampleSize; ++i){
		solverUpper->samples[i] = this->solver->samples[i];
		solverLower->samples[i] = this->solver->samples[i];
	}
	solverUpper->solve(1);
	solverLower->solve(1);
	this->QoI[0] = solverUpper->solve4QoI()-solverLower->solve4QoI();
}

// template <typename samplerType, typename solverType> 
// void MLChain_Uni_0i<samplerType, solverType>::runStep(std::default_random_engine* generator){
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
//         std::cout << "MCMCChain" << this->QoI[0] << std::endl;
// 	}
// 	this->acceptanceRate = this->acceptedNum/this->chainIdx;
// }

// template <typename samplerType, typename solverType> 
// void MLChain_Uni_0i<samplerType, solverType>::run(double QoImean[], std::default_random_engine* generator){
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
void MLChain_Uni_0i<samplerType, solverType>::generateIndSamples(int targetSampleSize, int rank, int l, std::default_random_engine* generator)
{
    std::string outputfile = "Uni_chain0";
    outputfile.append(std::to_string(l));
    outputfile.append("_ind_samples_");
    outputfile.append(std::to_string(rank));

    std::ifstream cFile(outputfile);
    std::ofstream myfile;

    int readCount=0;
    bool completed=false;
    double Workspace;
    double Workspace_Obs[20];
    int Workspace_Obs_size=10;
    double initialSample[this->sampleSize];
    this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));

    if (cFile.is_open()){
        std::string line;
        std::string line_back;
        getline(cFile, line);
        line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
        while(getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if (line[0] == '#' || line.empty()){
                continue;
            }
            if (line[0] == 'c'){
                completed=true;
                std::cout << outputfile << " is completed" << std::endl;
                break;
            }
            readCount++;
            line_back=line;
        }
        if (completed != true){
            auto delimiterPos = line_back.find_last_of(";");
            auto lastSample = line_back.substr(delimiterPos+1);
            std::cout << "last sample: " << lastSample << std::endl << std::flush;
            this->solver->samples[0] = std::stod(lastSample);
            std::cout << "continue from last sample" << this->solver->samples[0] << std::endl << std::flush;
        }
        cFile.close();
        myfile.open(outputfile, std::ios_base::app);
    } else {
        cFile.close();
        myfile.open(outputfile, std::ios_base::app);
        myfile<<"Obs,Likelihood,QoI,Samples"<<std::endl;

        this->solver->priorSample(initialSample);
        for (int i = 0; i < this->sampleSize; ++i){
            this->solver->samples[i] = initialSample[i];
            solverUpper->samples[i] = this->solver->samples[i];
            solverLower->samples[i] = this->solver->samples[i];
        }
        this->solver->solve();
        Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_size);
        myfile << Workspace_Obs[0];
        for (int i=1; i<20; i++){
            myfile<<"," << Workspace_Obs[i];
        }
        Workspace=this->solver->lnLikelihood();myfile<<";"<<Workspace;
        solverUpper->solve(1);
        solverLower->solve(1);
        Workspace=solverUpper->solve4QoI()-solverLower->solve4QoI();myfile<<";"<<Workspace;

        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<";"<<Workspace;
        }
        myfile << std::endl;
    }
    
    if (completed != true){
        targetSampleSize = targetSampleSize-readCount;
        for (int i = 1; i < targetSampleSize; ++i){
            this->sampleProposal();
            for (int i = 0; i < this->sampleSize; ++i){
                solverUpper->samples[i] = this->solver->samples[i];
                solverLower->samples[i] = this->solver->samples[i];
            }
            this->solver->solve();
            Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_size);
            myfile << Workspace_Obs[0];
            for (int i=1; i<20; i++){
                myfile<<"," << Workspace_Obs[i];
            }
            Workspace=this->solver->lnLikelihood();myfile<<";"<<Workspace;
            solverUpper->solve(1);
            solverLower->solve(1);
            Workspace=solverUpper->solve4QoI()-solverLower->solve4QoI();myfile<<";"<<Workspace;

            for (int i = 0; i<this->sampleSize; ++i){
                Workspace=this->solver->samples[i];myfile<<";"<<Workspace;
            }
            myfile << std::endl;
        }
        myfile << "complete";
    }
    myfile.close();
} 

template <typename samplerType, typename solverType> 
void MLChain_Uni_0i<samplerType, solverType>::runInd(double QoImean[], int startLocation, int rank, int l, std::default_random_engine* generator){
    double ObsInd[20];
    std::string samplesCSV="Uni_chain0";
    samplesCSV.append(std::to_string(l));
    samplesCSV.append("_ind_samples_");
    samplesCSV.append(std::to_string(rank));
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
    for (int i=0; i<19; ++i){
        getline(ss,temp,',');
        ObsInd[i]=std::stod(temp);
    }
    getline(ss, temp, ';');
    ObsInd[19]=std::stod(temp);

    getline(ss,temp,';');
    double Likelihoodt0Ind=std::stod(temp);
    double Likelihoodt1Ind=Likelihoodt0Ind;
    getline(ss,temp,';');
    double QoiIndt0=std::stod(temp);
    double QoiIndt1=QoiIndt0;


    this->chainIdx=1;	
    for (int i=this->chainIdx; i < this->numBurnin; ++i)
    {
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);

        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsInd[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsInd[19]=std::stod(temp);

        getline(ss,temp,';');
        Likelihoodt1Ind=std::stod(temp);
        getline(ss,temp,';');
        QoiIndt1=std::stod(temp);

        this->alpha = this->sampler->getAlpha(Likelihoodt0Ind, Likelihoodt1Ind);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            //std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0Ind=Likelihoodt1Ind;
            QoiIndt0=QoiIndt1;
        } else {
            //std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
    }
    this->QoIsum[0] = 0;
    for (int i = this->chainIdx; i<this->maxChainLength; ++i){
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);
  
        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsInd[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsInd[19]=std::stod(temp);
    
        getline(ss,temp,';');
        Likelihoodt1Ind=std::stod(temp);
        getline(ss,temp,';');
        QoiIndt1=std::stod(temp);
    
        //std::cout << "likelihood " << Likelihoodt1Ind << "qoit1 " << QoiIndt1 << std::endl;
    
        this->alpha = this->sampler->getAlpha(Likelihoodt0Ind, Likelihoodt1Ind);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            //std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0Ind=Likelihoodt1Ind;
            QoiIndt0=QoiIndt1;            
        } else {
            //std::cout << "sample rejected" << std::endl;
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
class MLChain_Uni_i0Upper : public MCMCChain<samplerType, solverType> {
public:
    double* QoI2;
    double* QoI2sum;
    double PhiUpper;
    double PhiLower;
   
    solverType* samplerLower;
    solverType* solver0;
   
    MLChain_Uni_i0Upper(int maxChainLength_, int sampleSize_, solverType* solver_, solverType* samplerLower_, solverType* solver0_) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_), samplerLower(samplerLower_), solver0(solver0_){
   	QoI2 = new double[sampleSize_];
   	QoI2sum = new double[sampleSize_];
    };
    ~MLChain_Uni_i0Upper(){
   	delete[] QoI2;
   	delete[] QoI2sum;
    };
   
    virtual void updateQoI();
    virtual void runStep(std::default_random_engine* generator);
    virtual void run(double QoImean[], double QoI2mean[], std::default_random_engine* generator);
    virtual void generateIndSamples(int targetSampleSize, int rank, int l, std::default_random_engine* generator);
    virtual void runInd(double QoImean[], double QoI2mean[], int startLocation, int rank, int l, std::default_random_engine* generator);
};

template <typename samplerType, typename solverType> 
void MLChain_Uni_i0Upper<samplerType, solverType>::updateQoI(){
	for (int i = 0; i < this->sampleSize; ++i){
		solver0->samples[i] = this->solver->samples[i];
		samplerLower->samples[i] = this->solver->samples[i];
	}
	solver0->solve(1);
	samplerLower->solve(0);
	this->PhiUpper = -this->lnLikelihoodt1;
	this->PhiLower = -samplerLower->lnLikelihood();
	this->QoI[0] = (1-exp(this->PhiUpper-this->PhiLower))*solver0->solve4QoI();
	QoI2[0] = (exp(this->PhiUpper-this->PhiLower)-1);
	//std::cout << "phiupper and lower: " << this->PhiUpper << " " << this->PhiLower << " " << this->QoI[0] << " " << QoI2[0] << std::endl;
}

template <typename samplerType, typename solverType> 
void MLChain_Uni_i0Upper<samplerType, solverType>::runStep(std::default_random_engine* generator){
	this->chainIdx += 1;
	this->sampleProposal();
	//std::cout << "m =" << this->solver->samples[0] << std::endl;
	this->solver->solve();
	this->updatelnLikelihood();
	this->checkAcceptance(generator);
	if (this->accepted == 1){
		this->lnLikelihoodt0 = this->lnLikelihoodt1;
		updateQoI();
		// std::cout << "accepted: ";
	} 
	// else {
	// 	// std::cout << "rejected: ";
	// }
	if (this->chainIdx > this->numBurnin){
		this->QoIsum[0] += this->QoI[0];
		QoI2sum[0] += QoI2[0];
	}
        //std::cout << QoI2[0] << " " << QoI2sum[0] << std::endl;
	this->acceptanceRate = this->acceptedNum/this->chainIdx;
}

template <typename samplerType, typename solverType> 
void MLChain_Uni_i0Upper<samplerType, solverType>::run(double QoImean[], double QoI2mean[], std::default_random_engine* generator){
    this->chainInit(generator);
    this->QoIsum[0] = 0;
    QoI2sum[0] = 0;
        //std::cout << "mac chain length" << this->maxChainLength << std::endl;
    for (int i = 1; i < this->maxChainLength+1; ++i){
	runStep(generator);
    }
    for (int i = 0; i < this->sampleSize; ++i){
	QoImean[i] = this->QoIsum[i]/(this->maxChainLength-this->numBurnin);
	QoI2mean[i] = QoI2sum[i]/(this->maxChainLength-this->numBurnin);
    }
    std::cout << "acceptance rate: " << this->acceptanceRate << std::endl;
    this->chainIdx = 0;
}


template <typename samplerType, typename solverType> 
void MLChain_Uni_i0Upper<samplerType, solverType>::generateIndSamples(int targetSampleSize, int rank, int l, std::default_random_engine* generator)
{
    std::string outputfile = "Uni_chain";
    outputfile.append(std::to_string(l));
    outputfile.append("0Upper_ind_samples_");
    outputfile.append(std::to_string(rank));

    std::ifstream cFile(outputfile);
    std::ofstream myfile;


    int readCount=0;
    bool completed=false;
    double Workspace;
    double Workspace_Obs[20];
    int Workspace_Obs_size=10;
    double initialSample[this->sampleSize];
    this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));

    if (cFile.is_open()){
        std::string line;
        std::string line_back;
        getline(cFile, line);
        line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
        while(getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if (line[0] == '#' || line.empty()){
                continue;
            }
            if (line[0] == 'c'){
                completed=true;
                std::cout << outputfile << " is completed" << std::endl;
                break;
            }
            readCount++;
            line_back=line;
        }
        if (completed != true){
            auto delimiterPos = line_back.find_last_of(";");
            auto lastSample = line_back.substr(delimiterPos+1);
            std::cout << "last sample: " << lastSample << std::endl << std::flush;
            this->solver->samples[0] = std::stod(lastSample);
            std::cout << "continue from last sample" << this->solver->samples[0] << std::endl << std::flush;
        }
        cFile.close();
        myfile.open(outputfile, std::ios_base::app);
    } else {
        cFile.close();
        myfile.open(outputfile, std::ios_base::app);
        myfile<<"ObsUpper,ObsLower,LikelihoodUpper,LikelihoodLower,QoI,samples"<<std::endl;

        this->solver->priorSample(initialSample);
        for (int i = 0; i < this->sampleSize; ++i){
            this->solver->samples[i] = initialSample[i];
            solver0->samples[i] = this->solver->samples[i];
            samplerLower->samples[i] = this->solver->samples[i];
        }

        this->solver->solve();
        samplerLower->solve();
        Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_size);
        myfile << Workspace_Obs[0];
        for (int i=1; i<20; i++){
            myfile<<"," << Workspace_Obs[i];
        }
        Workspace=samplerLower->solve4Obs(Workspace_Obs, Workspace_Obs_size);myfile<<";";
        myfile << Workspace_Obs[0];
        for (int i=1; i<20; i++){
            myfile<<"," << Workspace_Obs[i];
        }
        Workspace=this->solver->lnLikelihood();myfile<<";"<<Workspace;
        Workspace=samplerLower->lnLikelihood();myfile<<";"<<Workspace;
        solver0->solve(1);
        Workspace=solver0->solve4QoI();myfile<<";"<<Workspace;

        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<";"<<Workspace;
        }
        myfile << std::endl;
    }

    if (completed != true){
        targetSampleSize = targetSampleSize-readCount;
        for (int i = 1; i < targetSampleSize; ++i){
            this->sampleProposal();
            for (int i = 0; i < this->sampleSize; ++i){
                solver0->samples[i] = this->solver->samples[i];
                samplerLower->samples[i] = this->solver->samples[i];
            }
            this->solver->solve();
            samplerLower->solve();
            Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_size);
            myfile << Workspace_Obs[0];
            for (int i=1; i<20; i++){
                myfile<<"," << Workspace_Obs[i];
            }
            Workspace=samplerLower->solve4Obs(Workspace_Obs, Workspace_Obs_size);myfile<<";";
            myfile << Workspace_Obs[0];
            for (int i=1; i<20; i++){
                myfile<<"," << Workspace_Obs[i];
            }
            Workspace=this->solver->lnLikelihood();myfile<<";"<<Workspace;
            Workspace=samplerLower->lnLikelihood();myfile<<";"<<Workspace;
            solver0->solve(1);
            Workspace=solver0->solve4QoI();myfile<<";"<<Workspace;


            for (int i = 0; i<this->sampleSize; ++i){
                Workspace=this->solver->samples[i];myfile<<";"<<Workspace;
            }
            myfile << std::endl;
        }
        myfile << "complete";
    }
    myfile.close();
} 

template <typename samplerType, typename solverType> 
void MLChain_Uni_i0Upper<samplerType, solverType>::runInd(double A1mean[], double A2mean[], int startLocation, int rank, int l, std::default_random_engine* generator)
{
    double ObsIndUpper[20];
    double ObsIndLower[20];
    std::string samplesCSV="Uni_chain";
    samplesCSV.append(std::to_string(l));
    samplesCSV.append("0Upper_ind_samples_");
    samplesCSV.append(std::to_string(rank));
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

    for (int i=0; i<19; ++i){
        getline(ss,temp,',');
        ObsIndUpper[i]=std::stod(temp);
    }
    getline(ss, temp, ';');
    ObsIndUpper[19]=std::stod(temp);

    for (int i=0; i<19; ++i){
        getline(ss,temp,',');
        ObsIndLower[i]=std::stod(temp);
    }    
    getline(ss, temp, ';');
    ObsIndLower[19]=std::stod(temp);

    getline(ss,temp,';');
    double Likelihoodt0IndUpper=std::stod(temp);
    double Likelihoodt1IndUpper=Likelihoodt0IndUpper;
    getline(ss,temp,';');
    double Likelihoodt0IndLower=std::stod(temp);
    double Likelihoodt1IndLower=Likelihoodt0IndLower;    
    this->PhiLower=-Likelihoodt1IndLower;
    this->PhiUpper=-Likelihoodt1IndUpper;
    getline(ss,temp,';');
    double QoiIndt0=std::stod(temp);
    double QoiIndt1=QoiIndt0;


    this->chainIdx=1;
    for (int i=this->chainIdx; i < this->numBurnin; ++i)
    {
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);

        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsIndUpper[i]=std::stod(temp);
        }    
        getline(ss, temp, ';');
        ObsIndUpper[19]=std::stod(temp);

        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsIndLower[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsIndLower[19]=std::stod(temp);

        getline(ss,temp,';');
        Likelihoodt1IndUpper=std::stod(temp);
        getline(ss,temp,';');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,';');
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
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
    }

    this->QoI[0] = QoiIndt0;
	this->QoIsum[0] = 0;
	QoI2sum[0] = 0;
	for (int i = this->chainIdx; i<this->maxChainLength; ++i){
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);
        
        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsIndUpper[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsIndUpper[19]=std::stod(temp);

        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsIndLower[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsIndLower[19]=std::stod(temp);

        getline(ss,temp,';');
        Likelihoodt1IndUpper=std::stod(temp);
        getline(ss,temp,';');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,';');
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
            this->QoI[0] = QoiIndt0;
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
        this->QoIsum[0] += (1-exp(this->PhiUpper-this->PhiLower))*this->QoI[0];
        QoI2sum[0] += (exp(this->PhiUpper-this->PhiLower)-1);
	}
	for (int i = 0; i < this->sampleSize; ++i){
		A1mean[0] = this->QoIsum[0] / (this->maxChainLength-this->numBurnin);
		A2mean[0] = QoI2sum[0] / (this->maxChainLength-this->numBurnin);
	}
    indSamples.close();
}

template <typename samplerType, typename solverType> 
class MLChain_Uni_i0Lower : public MCMCChain<samplerType, solverType> {
public:
	solverType* solver0;

	MLChain_Uni_i0Lower(int maxChainLength_, int sampleSize_, solverType* solver_, solverType* solver0_) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_), solver0(solver0_){};
	~MLChain_Uni_i0Lower(){};

	virtual void updateQoI();
    virtual void generateIndSamples(int targetSampleSize, int rank, int l, std::default_random_engine* generator);
    virtual void runInd(double A0mean[], int startLocation, int rank, int l, std::default_random_engine* generator);
};

template <typename samplerType, typename solverType> 
void MLChain_Uni_i0Lower<samplerType, solverType>::updateQoI(){
	for (int i = 0; i < this->sampleSize; ++i){
		solver0->samples[i] = this->solver->samples[i];
	}
	solver0->solve(1);
	this->QoI[0] = solver0->solve4QoI();
}

template <typename samplerType, typename solverType> 
void MLChain_Uni_i0Lower<samplerType, solverType>::generateIndSamples(int targetSampleSize, int rank, int l, std::default_random_engine* generator)
{
    std::string outputfile = "Uni_chain";
    outputfile.append(std::to_string(l));
    outputfile.append("0Lower_ind_samples_");
    outputfile.append(std::to_string(rank));

    std::ifstream cFile(outputfile);
    std::ofstream myfile;

    int readCount=0;
    bool completed=false;
    double Workspace;
    double Workspace_Obs[20];
    int Workspace_Obs_size=10;
    double initialSample[this->sampleSize];
    this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));

    if (cFile.is_open()){
        std::string line;
        std::string line_back;
        getline(cFile, line);
        line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
        while(getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if (line[0] == '#' || line.empty()){
                continue;
            }
            if (line[0] == 'c'){
                completed=true;
                std::cout << outputfile << " is completed" << std::endl;
                break;
            }
            readCount++;
            line_back=line;
        }
        if (completed != true){
            auto delimiterPos = line_back.find_last_of(";");
            auto lastSample = line_back.substr(delimiterPos+1);
            std::cout << "last sample: " << lastSample << std::endl << std::flush;
            this->solver->samples[0] = std::stod(lastSample);
            std::cout << "continue from last sample" << this->solver->samples[0] << std::endl << std::flush;
        }
        cFile.close();
        myfile.open(outputfile, std::ios_base::app);
    } else {
        cFile.close();
        myfile.open(outputfile, std::ios_base::app);
        myfile<<"ObsLower,LikelihoodLower,QoI,samples"<<std::endl;

        this->solver->priorSample(initialSample);
        for (int i = 0; i < this->sampleSize; ++i){
            this->solver->samples[i] = initialSample[i];
            solver0->samples[i] = this->solver->samples[i];
        }
        this->solver->solve();
        Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_size);
        myfile << Workspace_Obs[0];
        for (int i=1; i<20; i++){
            myfile<<"," << Workspace_Obs[i];
        }

        Workspace=this->solver->lnLikelihood();myfile<<";"<<Workspace;
        solver0->solve(1);
            Workspace=solver0->solve4QoI();myfile<<";"<<Workspace;


        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<";"<<Workspace;
        }
        myfile << std::endl;
    }

    if (completed != true){
        targetSampleSize=targetSampleSize-readCount;
        for (int i = 1; i < targetSampleSize; ++i){
            this->sampleProposal();
            for (int i = 0; i < this->sampleSize; ++i){
                solver0->samples[i] = this->solver->samples[i];
            }
            this->solver->solve();
            Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_size);
            myfile << Workspace_Obs[0];
            for (int i=1; i<20; i++){
                myfile<<"," << Workspace_Obs[i];
            }
            Workspace=this->solver->lnLikelihood();myfile<<";"<<Workspace;
            solver0->solve(1);
            Workspace=solver0->solve4QoI();myfile<<";"<<Workspace;

            for (int i = 0; i<this->sampleSize; ++i){
                Workspace=this->solver->samples[i];myfile<<";"<<Workspace;
            }
            myfile << std::endl;
        }
        myfile << "complete";
    }
    myfile.close();
} 

template <typename samplerType, typename solverType> 
void MLChain_Uni_i0Lower<samplerType, solverType>::runInd(double A0mean[], int startLocation, int rank, int l, std::default_random_engine* generator)
{
    double ObsIndLower[20];
    std::string samplesCSV="Uni_chain";
    samplesCSV.append(std::to_string(l));
    samplesCSV.append("0Lower_ind_samples_");
    samplesCSV.append(std::to_string(rank));
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

    for (int i=0; i<19; ++i){
        getline(ss,temp,',');
        ObsIndLower[i]=std::stod(temp);
    }
    getline(ss, temp, ';');
    ObsIndLower[19]=std::stod(temp);

    getline(ss,temp,';');
    double Likelihoodt0IndLower=std::stod(temp);
    double Likelihoodt1IndLower=Likelihoodt0IndLower;  
    getline(ss,temp,';');
    double QoiIndt0=std::stod(temp);
    double QoiIndt1=QoiIndt0;

    this->chainIdx=1;
    for (int i=this->chainIdx; i < this->numBurnin; ++i)
    {
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);

        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsIndLower[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsIndLower[19]=std::stod(temp);

        getline(ss,temp,';');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,';');
        QoiIndt1=std::stod(temp);

        //std::cout << "likelihood: " << Likelihoodt1IndLower << " Qoi: " << QoiIndt1 <<std::endl;

        this->alpha = this->sampler->getAlpha(Likelihoodt0IndLower, Likelihoodt1IndLower);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndLower=Likelihoodt1IndLower;
            QoiIndt0=QoiIndt1;
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
    }

    this->QoI[0] = QoiIndt0;
    this->QoIsum[0] = 0;//I*(1-exp(this->PhiUpper-this->PhiLower))*this->QoI[0];

	for (int i = this->chainIdx; i<this->maxChainLength; ++i){
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);

        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsIndLower[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsIndLower[19]=std::stod(temp);

        getline(ss,temp,';');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,';');
        QoiIndt1=std::stod(temp);

        //std::cout << "likelihood: " << Likelihoodt1IndLower << " Qoi: " << QoiIndt1 <<std::endl;

        this->alpha = this->sampler->getAlpha(Likelihoodt0IndLower, Likelihoodt1IndLower);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            //std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndLower=Likelihoodt1IndLower;
            QoiIndt0=QoiIndt1;
            this->QoI[0] = QoiIndt0;
        } else {
            //std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
		this->QoIsum[0] += this->QoI[0];
        //std::cout << this->QoIsum[0] << " " << this->QoI[0] << std::endl;
	}
	for (int i = 0; i < this->sampleSize; ++i){
		A0mean[0] = this->QoIsum[0] / (this->maxChainLength-this->numBurnin);
	}
    indSamples.close();
}


template <typename samplerType, typename solverType> 
class MLChain_Uni_ijLower : public MCMCChain<samplerType, solverType> {
public:
	solverType* solverUpper;
	solverType* solverLower;

	MLChain_Uni_ijLower(int maxChainLength_, int sampleSize_, solverType* solver_, solverType* solverUpper_, solverType* solverLower_) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_), solverUpper(solverUpper_), solverLower(solverLower_){};
	~MLChain_Uni_ijLower(){};

	virtual void updateQoI();
        virtual void generateIndSamples(int targetSampleSize, int rank, int l1, int l2, std::default_random_engine* generator);
	virtual void runInd(double A0mean[], int startLocation, int rank, int l1, int l2, std::default_random_engine* generator);
};

template <typename samplerType, typename solverType> 
void MLChain_Uni_ijLower<samplerType, solverType>::updateQoI(){
	for (int i = 0; i < this->sampleSize; ++i){
		solverUpper->samples[i] = this->solver->samples[i];
		solverLower->samples[i] = this->solver->samples[i];
	}
	solverUpper->solve(1);
	solverLower->solve(1);
	this->QoI[0] = solverUpper->solve4QoI()-solverLower->solve4QoI();
}


template <typename samplerType, typename solverType> 
void MLChain_Uni_ijLower<samplerType, solverType>::generateIndSamples(int targetSampleSize, int rank, int l1, int l2, std::default_random_engine* generator)
{
    std::string outputfile = "Uni_chain";
    outputfile.append(std::to_string(l1));
    outputfile.append(std::to_string(l2));
    outputfile.append("Lower_ind_samples_");
    outputfile.append(std::to_string(rank));

    std::ifstream cFile(outputfile);
    std::ofstream myfile;

    int readCount=0;
    bool completed=false;
    double Workspace;
    double Workspace_Obs[20];
    int Workspace_Obs_size=10;
    double initialSample[this->sampleSize];
    this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));

    if (cFile.is_open()){
        std::string line;
        std::string line_back;
        getline(cFile, line);
        line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
        while(getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if (line[0] == '#' || line.empty()){
                continue;
            }
            if (line[0] == 'c'){
                completed=true;
                std::cout << outputfile << " is completed" << std::endl;
                break;
            }
            readCount++;
            line_back=line;
        }
        if (completed != true){
            auto delimiterPos = line_back.find_last_of(";");
            auto lastSample = line_back.substr(delimiterPos+1);
            std::cout << "last sample: " << lastSample << std::endl << std::flush;
            this->solver->samples[0] = std::stod(lastSample);
            std::cout << "continue from last sample" << this->solver->samples[0] << std::endl << std::flush;
        }
        cFile.close();
        myfile.open(outputfile, std::ios_base::app);
    } else {
        cFile.close();
        myfile.open(outputfile, std::ios_base::app);
        myfile<<"ObsLower,LikelihoodLower,QoI,samples"<<std::endl;

        this->solver->priorSample(initialSample);
        for (int i = 0; i < this->sampleSize; ++i){
            this->solver->samples[i] = initialSample[i];
            solverUpper->samples[i] = this->solver->samples[i];
            solverLower->samples[i] = this->solver->samples[i];
        }
        this->solver->solve();
        Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_size);
        myfile << Workspace_Obs[0];
        for (int i=1; i<20; i++){
            myfile<<"," << Workspace_Obs[i];
        }

        Workspace=this->solver->lnLikelihood();myfile<<";"<<Workspace;
        solverUpper->solve(1);
        solverLower->solve(1);
        Workspace=solverUpper->solve4QoI()-solverLower->solve4QoI();myfile<<";"<<Workspace;

        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<","<<Workspace;
        }
        myfile << std::endl;
    }

    if (completed != true){
        targetSampleSize=targetSampleSize-readCount;
	for (int i = 1; i < targetSampleSize; ++i){
            this->sampleProposal();
            for (int i = 0; i < this->sampleSize; ++i){
                solverUpper->samples[i] = this->solver->samples[i];
                solverLower->samples[i] = this->solver->samples[i];
            }
            this->solver->solve();
            Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_size);
            myfile << Workspace_Obs[0];
            for (int i=1; i<20; i++){
                myfile<<"," << Workspace_Obs[i];
            }

            Workspace=this->solver->lnLikelihood();myfile<<","<<Workspace;
            solverUpper->solve(1);
            solverLower->solve(1);
            Workspace=solverUpper->solve4QoI()-solverLower->solve4QoI();myfile<<","<<Workspace;


            for (int i = 0; i<this->sampleSize; ++i){
                Workspace=this->solver->samples[i];myfile<<","<<Workspace;
            }
            myfile << std::endl;
        }
        myfile << "complete";
    }
    myfile.close();
} 

template <typename samplerType, typename solverType> 
void MLChain_Uni_ijLower<samplerType, solverType>::runInd(double A0mean[], int startLocation, int rank, int l1, int l2, std::default_random_engine* generator)
{
    double ObsIndLower[20];
    std::string samplesCSV="Uni_chain";
    samplesCSV.append(std::to_string(l1));
    samplesCSV.append(std::to_string(l2));
    samplesCSV.append("Lower_ind_samples_");
    samplesCSV.append(std::to_string(rank));
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
    for (int i=0; i<19; ++i){
        getline(ss,temp,',');
        ObsIndLower[i]=std::stod(temp);
    }
    getline(ss, temp, ';');
    ObsIndLower[19]=std::stod(temp);

    // double ObsIndLower=std::stod(temp);
    getline(ss,temp,';');
    double Likelihoodt0IndLower=std::stod(temp);
    double Likelihoodt1IndLower=Likelihoodt0IndLower;    
    getline(ss,temp,';');
    double QoiIndt0=std::stod(temp);
    double QoiIndt1=QoiIndt0;


    this->chainIdx=1;
    for (int i=this->chainIdx; i < this->numBurnin; ++i)
    {
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);
        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsIndLower[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsIndLower[19]=std::stod(temp);

        getline(ss,temp,';');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,';');
        QoiIndt1=std::stod(temp);


        this->alpha = this->sampler->getAlpha(Likelihoodt0IndLower, Likelihoodt1IndLower);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndLower=Likelihoodt1IndLower;
            QoiIndt0=QoiIndt1;
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
    }

    this->QoI[0] = QoiIndt0;
	this->QoIsum[0] = 0;//I*(1-exp(this->PhiUpper-this->PhiLower))*this->QoI[0];

	for (int i = this->chainIdx; i<this->maxChainLength; ++i){
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);

        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsIndLower[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsIndLower[19]=std::stod(temp);

        getline(ss,temp,';');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,';');
        QoiIndt1=std::stod(temp);

        this->alpha = this->sampler->getAlpha(Likelihoodt0IndLower, Likelihoodt1IndLower);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndLower=Likelihoodt1IndLower;
            QoiIndt0=QoiIndt1;
            this->QoI[0] = QoiIndt0;
        } else {
            // std::cout << "sample rejected" << std::endl;
            this->accepted = 0;
        }
		this->QoIsum[0] += this->QoI[0];
	}
	for (int i = 0; i < this->sampleSize; ++i){
		A0mean[0] = this->QoIsum[0] / (this->maxChainLength-this->numBurnin);
	}
    indSamples.close();
}

template <typename samplerType, typename solverType> 
class MLChain_Uni_ijUpper : public MCMCChain<samplerType, solverType> {
public:
	double* QoI2;
	double* QoI2sum;
	double PhiUpper;
	double PhiLower;

	solverType* samplerLower;
	solverType* solverUpper;
	solverType* solverLower;

	MLChain_Uni_ijUpper(int maxChainLength_, int sampleSize_, solverType* solver_, solverType* samplerLower_, solverType* solverUpper_, solverType* solverLower_) : MCMCChain<samplerType, solverType>(maxChainLength_, sampleSize_, solver_), samplerLower(samplerLower_), solverUpper(solverUpper_), solverLower(solverLower_){
		QoI2 = new double[sampleSize_];
		QoI2sum = new double[sampleSize_];
	};
	~MLChain_Uni_ijUpper(){
		delete[] QoI2;
		delete[] QoI2sum;
	};

	virtual void updateQoI();
	virtual void runStep(std::default_random_engine* generator);
	virtual void run(double QoImean[], double QoI2mean[], std::default_random_engine* generator);
    virtual void generateIndSamples(int targetSampleSize, int rank, int l1, int l2, std::default_random_engine* generator);
	virtual void runInd(double A1mean[], double A2mean[], int startLocation, int rank, int l1, int l2, std::default_random_engine* generator);
};

template <typename samplerType, typename solverType> 
void MLChain_Uni_ijUpper<samplerType, solverType>::updateQoI(){
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
	this->QoI[0] = (1-exp(PhiUpper-PhiLower))*(solverUpper->solve4QoI()-solverLower->solve4QoI());
	QoI2[0] = (exp(PhiUpper-PhiLower)-1);
}

template <typename samplerType, typename solverType> 
void MLChain_Uni_ijUpper<samplerType, solverType>::runStep(std::default_random_engine* generator){
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
		this->QoIsum[0] += this->QoI[0];
		QoI2sum[0] += QoI2[0];
	}
	this->acceptanceRate = this->acceptedNum/this->chainIdx;
}

template <typename samplerType, typename solverType> 
void MLChain_Uni_ijUpper<samplerType, solverType>::run(double QoImean[], double QoI2mean[], std::default_random_engine* generator){
	this->chainInit(generator);
	this->QoIsum[0] = 0;
	QoI2sum[0] = 0;
	for (int i = 1; i < this->maxChainLength+1; ++i){
		runStep(generator);
	}
	for (int i = 0; i < this->sampleSize; ++i){
		QoImean[i] = this->QoIsum[i]/(this->maxChainLength-this->numBurnin);
		QoI2mean[i] = QoI2sum[i]/(this->maxChainLength-this->numBurnin);
	}
}


template <typename samplerType, typename solverType> 
void MLChain_Uni_ijUpper<samplerType, solverType>::generateIndSamples(int targetSampleSize, int rank, int l1, int l2, std::default_random_engine* generator)
{
    std::string outputfile = "Uni_chain";
    outputfile.append(std::to_string(l1));
    outputfile.append(std::to_string(l2));
    outputfile.append("Upper_ind_samples_");
    outputfile.append(std::to_string(rank));

    std::ifstream cFile(outputfile);
    std::ofstream myfile;


    int readCount=0;
    bool completed=false;
    double Workspace;
    double Workspace_Obs[20];
    int Workspace_Obs_size=10;
    double initialSample[this->sampleSize];
    this->solver->updateGeneratorSeed(1000*this->uniform_distribution(*generator));


    if (cFile.is_open()){
        std::string line;
        std::string line_back;
        getline(cFile, line);
        line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
        while(getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if (line[0] == '#' || line.empty()){
                continue;
            }
            if (line[0] == 'c'){
                completed=true;
                std::cout << outputfile << " is completed" << std::endl;
                break;
            }
            readCount++;
            line_back=line;
        }
        if (completed != true){
            auto delimiterPos = line_back.find_last_of(";");
            auto lastSample = line_back.substr(delimiterPos+1);
            std::cout << "last sample: " << lastSample << std::endl << std::flush;
            this->solver->samples[0] = std::stod(lastSample);
            std::cout << "continue from last sample" << this->solver->samples[0] << std::endl << std::flush;
        }
        cFile.close();
        myfile.open(outputfile, std::ios_base::app);
    } else {
        cFile.close();
        myfile.open(outputfile, std::ios_base::app);
        myfile<<"ObsUpper,ObsLower,LikelihoodUpper,LikelihoodLower,QoI,samples"<<std::endl;

	this->solver->priorSample(initialSample);
	for (int i = 0; i < this->sampleSize; ++i){
		this->solver->samples[i] = initialSample[i];
		solverUpper->samples[i] = this->solver->samples[i];
        solverLower->samples[i] = this->solver->samples[i];
		samplerLower->samples[i] = this->solver->samples[i];
	}
	this->solver->solve();
	samplerLower->solve();
        Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_size);
        myfile << Workspace_Obs[0];
        for (int i=1; i<20; i++){
            myfile<<"," << Workspace_Obs[i];
        }
        Workspace=samplerLower->solve4Obs(Workspace_Obs, Workspace_Obs_size);myfile<<";";
        myfile << Workspace_Obs[0];
        for (int i=1; i<20; i++){
            myfile<<"," << Workspace_Obs[i];
        }

        Workspace=this->solver->lnLikelihood();myfile<<";"<<Workspace;
        Workspace=samplerLower->lnLikelihood();myfile<<";"<<Workspace;
        solverUpper->solve(1);
        solverLower->solve(1);
        Workspace=solverUpper->solve4QoI()-solverLower->solve4QoI();myfile<<";"<<Workspace;

        for (int i = 0; i<this->sampleSize; ++i){
            Workspace=this->solver->samples[i];myfile<<";"<<Workspace;
        }
        myfile << std::endl;
    }

    if (completed != true){
	for (int i = 1; i < targetSampleSize; ++i){
            this->sampleProposal();
            for (int i = 0; i < this->sampleSize; ++i){
                solverUpper->samples[i] = this->solver->samples[i];
                solverLower->samples[i] = this->solver->samples[i];
        		samplerLower->samples[i] = this->solver->samples[i];
            }
            this->solver->solve();
            samplerLower->solve();
            Workspace=this->solver->solve4Obs(Workspace_Obs, Workspace_Obs_size);
            myfile << Workspace_Obs[0];
            for (int i=1; i<20; i++){
                myfile<<"," << Workspace_Obs[i];
            }

            Workspace=samplerLower->solve4Obs(Workspace_Obs, Workspace_Obs_size);myfile<<";";
            myfile << Workspace_Obs[0];
            for (int i=1; i<20; i++){
                myfile<<"," << Workspace_Obs[i];
            }

            Workspace=this->solver->lnLikelihood();myfile<<";"<<Workspace;
            Workspace=samplerLower->lnLikelihood();myfile<<";"<<Workspace;
            solverUpper->solve(1);
            solverLower->solve(1);
            Workspace=solverUpper->solve4QoI()-solverLower->solve4QoI();myfile<<";"<<Workspace;


            for (int i = 0; i<this->sampleSize; ++i){
                Workspace=this->solver->samples[i];myfile<<";"<<Workspace;
            }
            myfile << std::endl;
        }
        myfile << "complete";
    }
    myfile.close();
} 


template <typename samplerType, typename solverType> 
void MLChain_Uni_ijUpper<samplerType, solverType>::runInd(double A1mean[], double A2mean[], int startLocation, int rank, int l1, int l2, std::default_random_engine* generator)
{
    double ObsIndUpper[20];
    double ObsIndLower[20];
    std::string samplesCSV="Uni_chain";
    samplesCSV.append(std::to_string(l1));
    samplesCSV.append(std::to_string(l2));
    samplesCSV.append("Upper_ind_samples_");
    samplesCSV.append(std::to_string(rank));
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

    for (int i=0; i<19; ++i){
        getline(ss,temp,',');
        ObsIndUpper[i]=std::stod(temp);
    }
    getline(ss, temp, ';');
    ObsIndUpper[19]=std::stod(temp);

    for (int i=0; i<19; ++i){
        getline(ss,temp,',');
        ObsIndLower[i]=std::stod(temp);
    }
    getline(ss, temp, ';');
    ObsIndLower[19]=std::stod(temp);

    getline(ss,temp,';');
    double Likelihoodt0IndUpper=std::stod(temp);
    double Likelihoodt1IndUpper=Likelihoodt0IndUpper;
    getline(ss,temp,';');
    double Likelihoodt0IndLower=std::stod(temp);
    double Likelihoodt1IndLower=Likelihoodt0IndLower;    
    this->PhiLower=-Likelihoodt1IndLower;
    this->PhiUpper=-Likelihoodt1IndUpper;	
    getline(ss,temp,';');
    double QoiIndt0=std::stod(temp);
    double QoiIndt1=QoiIndt0;


    this->chainIdx=1;
    for (int i=this->chainIdx; i < this->numBurnin; ++i)
    {
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);

        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsIndUpper[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsIndUpper[19]=std::stod(temp);
    
        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsIndLower[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsIndLower[19]=std::stod(temp);

        getline(ss,temp,';');
        Likelihoodt1IndUpper=std::stod(temp);
        getline(ss,temp,';');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,';');
        QoiIndt1=std::stod(temp);

        this->alpha = this->sampler->getAlpha(Likelihoodt0IndUpper, Likelihoodt1IndUpper);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted (burn-in period)" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndUpper=Likelihoodt1IndUpper;
            QoiIndt0=QoiIndt1;
            this->PhiLower=-Likelihoodt1IndLower;
            this->PhiUpper=-Likelihoodt1IndUpper;
        } else {
            // std::cout << "sample rejected (burn-in period)" << std::endl;
            this->accepted = 0;
        }
    }

    this->QoI[0] = QoiIndt0;
    this->QoIsum[0] = 0;//I*(1-exp(this->PhiUpper-this->PhiLower))*this->QoI[0];
    QoI2sum[0] = 0;//I*(exp(this->PhiUpper-this->PhiLower)-1);


    for (int i = this->chainIdx; i<this->maxChainLength; ++i){
        this->chainIdx++;
        getline(indSamples,line);
        ss.str(line);
        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsIndUpper[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsIndUpper[19]=std::stod(temp);
    
        for (int i=0; i<19; ++i){
            getline(ss,temp,',');
            ObsIndLower[i]=std::stod(temp);
        }
        getline(ss, temp, ';');
        ObsIndLower[19]=std::stod(temp);

        getline(ss,temp,';');
        Likelihoodt1IndUpper=std::stod(temp);
        getline(ss,temp,';');
        Likelihoodt1IndLower=std::stod(temp);
        getline(ss,temp,';');
        QoiIndt1=std::stod(temp);

        this->alpha = this->sampler->getAlpha(Likelihoodt0IndUpper, Likelihoodt1IndUpper);
        this->alphaUni = log(this->uniform_distribution(*generator));
        if (this->alphaUni < this->alpha){
            // std::cout << "sample accpeted (sampling period)" << std::endl;
            this->accepted = 1;
            Likelihoodt0IndUpper=Likelihoodt1IndUpper;
            QoiIndt0=QoiIndt1;
            this->PhiLower=-Likelihoodt1IndLower;
            this->PhiUpper=-Likelihoodt1IndUpper;
            this->QoI[0] = QoiIndt0;
        } else {
            // std::cout << "sample rejected (sampling period)" << std::endl;
            this->accepted = 0;
        }
        this->QoIsum[0] += (1-exp(PhiUpper-PhiLower))*this->QoI[0];
        QoI2sum[0] += (exp(PhiUpper-PhiLower)-1);
	}
	for (int i = 0; i < this->sampleSize; ++i){
		A1mean[0] = this->QoIsum[0] / (this->maxChainLength-this->numBurnin);
		A2mean[0] = QoI2sum[0] / (this->maxChainLength-this->numBurnin);
	}
    indSamples.close();
}
