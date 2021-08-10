#include "dataIO.h"

void write2txt(double array[], int arraySize, std::string pathName){
	std::ofstream myfile;
	myfile.open(pathName);
	for (int i = 0; i < arraySize; ++i){
		myfile << array[i] << " ";
	}	
	myfile.close();
};

void write2txt(double array1[], double array2[], int arraySize, std::string pathName){
	std::ofstream myfile;
	myfile.open(pathName);
	for (int i = 0; i < arraySize; ++i){
		myfile << array1[i] << " ";
	}
	myfile << std::endl;
	for (int i = 0; i < arraySize; ++i){
		myfile << array2[i] << " ";
	}
	myfile.close();
};


void write2csv(double array[], std::string pathName){

};

void write2csv(double array1[], double array2[], std::string pathName){

};

void txt2read(double array[], int arraySize, std::string pathName){
	std::ifstream myfile;
	myfile.open(pathName, std::ios_base::in);
	for(int i = 0; i < arraySize; ++i){
		myfile >> array[i];
	}
	myfile.close();
}

void txt2read(double array1[], double array2[], int arraySize, std::string pathName){
	std::ifstream myfile;
	myfile.open(pathName, std::ios_base::in);
	for (int i = 0; i < arraySize; i++){
		myfile >> array1[i];
	}
	for (int i = 0; i < arraySize; i++){
		myfile >> array2[i];
	}
	myfile.close();
};
