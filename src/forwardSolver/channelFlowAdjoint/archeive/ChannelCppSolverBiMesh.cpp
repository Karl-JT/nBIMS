#include "ChannelCppSolverBiMesh.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

#include <adolc/adolc.h>

void meshGen(int l, int n0, double yCoordinate[]){
    // Node Coordinate Initialization
    int m = n0*pow(2,l+1)+1;
    double temp[m];
    yCoordinate[0] = 0;
    yCoordinate[n0] = 1;
    for (int i=1; i < n0; i++){
        yCoordinate[i] = 1.0/pow(2, n0)*pow(2, i);
        memcpy(temp, yCoordinate, sizeof(double)*(n0*pow(2, l+1)+1));
    }
    for (int j=1; j < l+1; j++){
        for (int i=0; i < pow(2, j-1)*n0; i++){
            yCoordinate[2*i] = temp[i];
            yCoordinate[2*i+1] = (temp[i]+temp[i+1])/2.0;
        }
        yCoordinate[(int) pow(2, j)*n0] = temp[(int) pow(2, j-1)*n0];
        memcpy(temp, yCoordinate, sizeof(double)*(n0*pow(2, l+1)+1));
    }
    for (int i=0; i < n0*pow(2, l+1)+1; i++){
        if (i < n0*pow(2, l)+1){
            yCoordinate[i] = yCoordinate[i]-1;
        }else{
            yCoordinate[i] = -yCoordinate[(int) (n0*pow(2, l+1)-i)];
        }
    }
}

