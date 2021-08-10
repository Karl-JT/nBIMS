#include "ChannelCppSolverGeoMesh.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

#include <adolc/adolc.h>

void meshGen(int simpleGridingRatio, int m, double reTau, double yCoordinate[]){
    // Node Coordinate Initialization
    double gridRatio = pow(simpleGridingRatio, 1.0/((m+1)/2.0 - 2));
    double firstNodeDist = (1-gridRatio) / (1-pow(gridRatio, (m+1)/2.0 - 1));
    double tempGridSize = firstNodeDist;
    yCoordinate[0] = -1;
    for (int i = 1; i < (m+1)/2.0-1; i++) {
            yCoordinate[i] = yCoordinate[i - 1] + tempGridSize;
            tempGridSize = tempGridSize * gridRatio;
    }
    yCoordinate[(m+1)/2] = 0;
    for (int i = (m+1)/2.0; i < m; i++){
            yCoordinate[i] = -yCoordinate[m-1-i];
    }
    cout << "simpleGridingRatio: " << simpleGridingRatio << endl;
}
