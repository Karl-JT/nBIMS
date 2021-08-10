#include "ChannelCppSolverBiMesh.h"


void caseProp::meshGen(int l, int n0, double yCoordinate[]){
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


void caseProp::caseInitialize(int l_, double reTau){
    //initialize reTau number and update corresponding properties
    if (reTau == 180) {
        reTau            = reTau;
        deltaTime        = 1;
        nu               = 3.4e-4;
        frictionVelocity = 6.37309e-2;
    }
    if (reTau == 550) {
        reTau            = reTau;
        deltaTime        = 1;
        nu               = 1e-4;
        frictionVelocity = 5.43496e-2;
    }
    if (reTau == 1000) {
        reTau            = reTau;
        deltaTime        = 1;
        nu               = 5e-5;
        frictionVelocity = 5.00256e-2;
    }
    if (reTau == 2000) {
        reTau            = reTau;
        deltaTime        = 1;
        nu               = 2.3e-5;
        frictionVelocity = 4.58794e-2;
    }
    if (reTau == 5200) {
        reTau            = reTau;
        deltaTime        = 1;
        nu               = 8e-6;
        frictionVelocity =4.14872e-2;
    }
    //update mesh dimensions according to reTau
    n0 = 9;
    l = l_; //multi-level, start with l0
    cut_off = 500000.0;// /pow(2.0, l_);
    m = n0*pow(2, l+1)+1; //total grid points
    yCoordinate   = std::make_unique<double[]>(m);
    dnsData       = std::make_unique<double[]>(m);
    initXVelocity = std::make_unique<double[]>(m);
    nodeWeight    = std::make_unique<double[]>(m);
    pressure      = std::make_unique<double[]>(m);
    xVelocity         = std::make_unique<adouble[]>(m);
    xVelocityGradient = std::make_unique<adouble[]>(m);
    k                 = std::make_unique<adouble[]>(m);
    omega             = std::make_unique<adouble[]>(m);
    nut               = std::make_unique<adouble[]>(m);
    betaML            = std::make_unique<adouble[]>(m);
    R                 = std::make_unique<adouble[]>(3*m);
    solution          = std::make_unique<adouble[]>(3*m);


    for (int i = 0; i < m; i++){
        pressure[i]      = pow(frictionVelocity, 2);
        xVelocity[i]     = 0;
        initXVelocity[i] = 0;
        k[i]             = 1e-8;
        omega[i]         = 1e5;
        nut[i]           = 1e-5;
        betaML[i]        = 1;
    }

    for (int i = 0; i < 3*m; i++){
        R[i]        = 1;
        solution[i] = 0;
    }
    //update mesh data on yCoordinate
    meshGen(l, n0, yCoordinate.get());
    nodeWeight[0] = (yCoordinate[1]-yCoordinate[0])/2;
    for (int i = 1; i < m-1; i++){
        nodeWeight[i] = (yCoordinate[i+1] - yCoordinate[i-1])/2;
    }
    nodeWeight[m-1] = (yCoordinate[m-1] - yCoordinate[m-2])/2;

    //update interpreted DNS data
    dnsDataInterpreter dnsDataTable(yCoordinate.get(), reTau, m);
    memcpy(dnsData.get(), dnsDataTable.U, sizeof(double)*m);
}

void caseProp::thomasSolver(adouble vectorA[], adouble vectorB[], adouble vectorC[], adouble vectorD[], adouble solution[], int vectorSize){
    adouble newVectorC[vectorSize];
    adouble newVectorD[vectorSize];

    for (int i = 0; i < vectorSize; i ++){
        newVectorC[i] = 0;
        newVectorD[i] = 0;
    }

    newVectorC[0] = vectorC[0] / vectorB[0];
    newVectorD[0] = vectorD[0] / vectorB[0];

    //#pragma acc kernels
    for (int i = 1; i < vectorSize - 1; i++) {
        newVectorC[i] = vectorC[i] / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
        newVectorD[i] = (vectorD[i] - vectorA[i] * newVectorD[i - 1]) / (vectorB[i] - vectorA[i] * newVectorC[i - 1]);
    }
    newVectorD[vectorSize - 1] = (vectorD[vectorSize - 1] - vectorA[vectorSize - 1] * newVectorD[vectorSize - 2]) / (vectorB[vectorSize - 1] - vectorA[vectorSize - 1] * newVectorC[vectorSize - 2]);

    solution[vectorSize - 1] = newVectorD[vectorSize - 1];
    for (int i = 0; i < vectorSize - 1; i++) {
        solution[vectorSize - 2 - i] = newVectorD[vectorSize - 2 - i] - newVectorC[vectorSize - 2 - i] * solution[vectorSize - 1 - i];
    }
}

void caseProp::linearDiscretization(adouble vectorA[], adouble vectorB[], adouble vectorC[], adouble vectorD[], int stepCount){
    double scaleFactor = 1.01;
    
    for (int i = 0; i < m; i++){
        nut[i] = k[i]/omega[i] * betaML[i];
    }
    //discretizatin of flow equation
    vectorB[0]     = -1;
    vectorC[0]     = 0;
    vectorA[m - 1] = 0;
    vectorB[m - 1] = -1;
    vectorD[0]     = boundaryVelocity[0]; //boundaryVelocity
    vectorD[m - 1] = boundaryVelocity[1]; //boundaryVelocity

    //#pragma acc parallel loop
    for (int i = 1; i < m - 1; i++) {
        vectorA[i] =  2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])* (nu + nut[i - 1] / 2.0 + nut[i] / 2.0) / (yCoordinate[i] - yCoordinate[i - 1]);
        vectorB[i] = -2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*((nu + nut[i] / 2.0 + nut[i + 1] / 2.0) / (yCoordinate[i + 1] - yCoordinate[i]) + (nu + nut[i - 1] / 2.0 + nut[i] / 2.0) / (yCoordinate[i] - yCoordinate[i - 1])) - 1/(deltaTime*pow(scaleFactor, stepCount));
        vectorC[i] =  2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])* (nu + nut[i] / 2.0 + nut[i + 1] / 2.0) / (yCoordinate[i + 1] - yCoordinate[i]);
        vectorD[i] = -pressure[i] - xVelocity[i]/(deltaTime*pow(scaleFactor, stepCount));
    }

    //discretizatin of omega equation
    double sigma = 0.5;
    double alpha = 3.0 / 40.0;
    double gamma = 5.0 / 9.0;
    int boundaryPoints = 3;

    vectorB[m] = -1;
    vectorC[m] = 0;
    vectorD[m] = -1e100;

    for (int i = 1; i < boundaryPoints; i++) {
        vectorA[i+m] = 0;
        vectorB[i+m] = -1;
        vectorC[i+m] = 0;
        vectorD[i+m] = -6.0 * nu / (0.00708 * pow(yCoordinate[i]+1, 2));
    }

    vectorA[2*m-1] = 1;
    vectorB[2*m-1] = -1;
    vectorD[2*m-1] = -1e100;

    for (int i = m-boundaryPoints; i < m-1; i++) {
        vectorA[i+m] = 0;
        vectorB[i+m] = -1;
        vectorC[i+m] = 0;
        vectorD[i+m] = -6.0 * nu / (0.00708 * pow(1-yCoordinate[i], 2));
    }

    //#pragma acc parallel loop
    for (int i = boundaryPoints; i < m-boundaryPoints; i++) {
        vectorA[i+m] =  2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(nu + sigma * (nut[i] / 2.0 + nut[i - 1] / 2.0)) / (yCoordinate[i] - yCoordinate[i - 1]);
        vectorB[i+m] = -alpha * omega[i] - 2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*((nu + sigma * (nut[i] / 2.0 + nut[i - 1] / 2.0)) / (yCoordinate[i] - yCoordinate[i - 1]) + (nu + sigma * (nut[i] / 2.0 + nut[i + 1] / 2.0)) / (yCoordinate[i + 1] - yCoordinate[i])) - 1/(deltaTime*pow(scaleFactor, stepCount));
        vectorC[i+m] =  2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(nu + sigma * (nut[i] / 2.0 + nut[i + 1] / 2.0)) / (yCoordinate[i + 1] - yCoordinate[i]);
        vectorD[i+m] = -gamma * pow(((xVelocity[i + 1] - xVelocity[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1])), 2.0) - omega[i]/(deltaTime*pow(scaleFactor, stepCount)); // - max(0, sigmaD / omega[i] * ((k[i + 1] - k[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]))*((omega[i + 1] - omega[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]))))*deltaTime - omega[i];
    }


    //discretizatin of k equation
    double sigmaStar = 0.5;
    double alphaStar = 0.09;

    vectorB[2*m] = -1.0;
    vectorC[2*m] = 0.0;
    vectorA[3*m - 1] = 0;
    vectorB[3*m - 1] = -1.0;
    vectorD[2*m] = 0;
    vectorD[3*m - 1] = 0;

    //#pragma acc parallel loop
    for (int i = 1; i < m - 1; i++){
        vectorA[i+2*m] = 2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(nu + sigmaStar * (nut[i] / 2 + nut[i - 1] / 2)) / (yCoordinate[i] - yCoordinate[i - 1]);
        vectorB[i+2*m] = -alphaStar * omega[i] - 2 / (yCoordinate[i + 1] - yCoordinate[i - 1])*((nu + sigmaStar * (nut[i] / 2 + nut[i - 1] / 2)) / (yCoordinate[i] - yCoordinate[i - 1]) + (nu + sigmaStar * (nut[i] / 2 + nut[i + 1] / 2)) / (yCoordinate[i + 1] - yCoordinate[i])) - 1/(deltaTime*pow(scaleFactor, stepCount));
        vectorC[i+2*m] = 2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(nu + sigmaStar * (nut[i] / 2 + nut[i + 1] / 2)) / (yCoordinate[i + 1] - yCoordinate[i]);
        vectorD[i+2*m] = -nut[i] * pow((xVelocity[i + 1] - xVelocity[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]), 2.0) - k[i]/(deltaTime*pow(scaleFactor, stepCount));
    }
}


void caseProp::flowSolver(int stepCount) {
    //initiate linear system
    adouble vectorA[3*m];
    adouble vectorB[3*m];
    adouble vectorC[3*m];
    adouble vectorD[3*m];

    for (int i = 0; i < 3*m; i ++){
        vectorA[i] = 0;
        vectorB[i] = 0;
        vectorC[i] = 0;
        vectorD[i] = 0;
    }
    linearDiscretization(vectorA, vectorB, vectorC, vectorD, stepCount);
    thomasSolver(vectorA, vectorB, vectorC, vectorD, solution.get(), 3*m);

    for (int i = 0; i < m; i ++){
        xVelocity[i] = solution[i];
        omega[i] = solution[m + i];
        k[i] = solution[2*m + i];
    }
};

void caseProp::residualUpdate(){
    //initiate linear system
    adouble vectorA[3*m];
    adouble vectorB[3*m];
    adouble vectorC[3*m];
    adouble vectorD[3*m];

    for (int i = 0; i < 3*m; i ++){
        vectorA[i] = 0;
        vectorB[i] = 0;
        vectorC[i] = 0;
        vectorD[i] = 0;
    }

    //linearization
    for (int i = 0; i < m; i++){
        nut[i] = betaML[i]*k[i]/omega[i];
    }
    //discretizatin of flow equation
    vectorB[0] = -1;
    vectorC[0] = 0;
    vectorA[m - 1] = 0;
    vectorB[m - 1] = -1;
    vectorD[0] = boundaryVelocity[0]; //boundaryVelocity
    vectorD[m - 1] = boundaryVelocity[1]; //boundaryVelocity

    //#pragma acc parallel loop
    for (int i = 1; i < m - 1; i++) {
        vectorA[i] = 2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(nu + nut[i - 1] / 2.0 + nut[i] / 2.0) / (yCoordinate[i] - yCoordinate[i - 1]);
        vectorB[i] = -2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*((nu + nut[i] / 2.0 + nut[i + 1] / 2.0) / (yCoordinate[i + 1] - yCoordinate[i]) + (nu + nut[i - 1] / 2.0 + nut[i] / 2.0) / (yCoordinate[i] - yCoordinate[i - 1]));
        vectorC[i] = 2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(nu + nut[i] / 2.0 + nut[i + 1] / 2.0) / (yCoordinate[i + 1] - yCoordinate[i]);
        vectorD[i] = -pressure[i];
    }

    //discretizatin of omega equation
    double sigma = 0.5;
    double alpha = 3.0 / 40.0;
    double gamma = 5.0 / 9.0;
    int boundaryPoints = 3;

    vectorB[m] = -1;
    vectorC[m] = 0;
    vectorD[m] = -1e100;

    for (int i = 1; i < boundaryPoints; i++) {
        vectorA[i+m] = 0;
        vectorB[i+m] = -1;
        vectorC[i+m] = 0;
        vectorD[i+m] = -6.0 * nu / (0.00708 * pow(yCoordinate[i]+1, 2));
    }

    vectorA[2*m-1] = 1;
    vectorB[2*m-1] = -1;
    vectorD[2*m-1] = -1e100;

    for (int i = m-boundaryPoints; i < m-1; i++) {
        vectorA[i+m] = 0;
        vectorB[i+m] = -1;
        vectorC[i+m] = 0;
        vectorD[i+m] = -6.0 * nu / (0.00708 * pow(1-yCoordinate[i], 2));
    }

    //#pragma acc parallel loop
    for (int i = boundaryPoints; i < m-boundaryPoints; i++) {
        vectorA[i+m] = 2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(nu + sigma * (nut[i] / 2.0 + nut[i - 1] / 2.0)) / (yCoordinate[i] - yCoordinate[i - 1]);
        vectorB[i+m] = -alpha * omega[i] - 2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*((nu + sigma * (nut[i] / 2.0 + nut[i - 1] / 2.0)) / (yCoordinate[i] - yCoordinate[i - 1]) + (nu + sigma * (nut[i] / 2.0 + nut[i + 1] / 2.0)) / (yCoordinate[i + 1] - yCoordinate[i]));
        vectorC[i+m] = 2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(nu + sigma * (nut[i] / 2.0 + nut[i + 1] / 2.0)) / (yCoordinate[i + 1] - yCoordinate[i]);
        vectorD[i+m] = -gamma * pow(((xVelocity[i + 1] - xVelocity[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1])), 2.0); // - max(0, sigmaD / omega[i] * ((k[i + 1] - k[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]))*((omega[i + 1] - omega[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]))))*deltaTime - omega[i];
    }


    //discretizatin of k equation
    double sigmaStar = 0.5;
    double alphaStar = 0.09;

    vectorB[2*m] = -1.0;
    vectorC[2*m] = 0.0;
    vectorA[3*m - 1] = 0;
    vectorB[3*m - 1] = -1.0;
    vectorD[2*m] = 0;
    vectorD[3*m - 1] = 0;

    //#pragma acc parallel loop
    for (int i = 1; i < m - 1; i++){
        vectorA[i+2*m] = 2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(nu + sigmaStar * (nut[i] / 2 + nut[i - 1] / 2)) / (yCoordinate[i] - yCoordinate[i - 1]);
        vectorB[i+2*m] = -alphaStar * omega[i] - 2 / (yCoordinate[i + 1] - yCoordinate[i - 1])*((nu + sigmaStar * (nut[i] / 2 + nut[i - 1] / 2)) / (yCoordinate[i] - yCoordinate[i - 1]) + (nu + sigmaStar * (nut[i] / 2 + nut[i + 1] / 2)) / (yCoordinate[i + 1] - yCoordinate[i]));
        vectorC[i+2*m] = 2.0 / (yCoordinate[i + 1] - yCoordinate[i - 1])*(nu + sigmaStar * (nut[i] / 2 + nut[i + 1] / 2)) / (yCoordinate[i + 1] - yCoordinate[i]);
        vectorD[i+2*m] = -nut[i] * pow((xVelocity[i + 1] - xVelocity[i - 1]) / (yCoordinate[i + 1] - yCoordinate[i - 1]), 2.0);
    }
    //end of linearization

    R[0] = 0;
    for (int i = 1; i < m-1; i++){
        R[i] = vectorA[i]*xVelocity[i-1]+vectorB[i]*xVelocity[i]+vectorC[i]*xVelocity[i+1]-vectorD[i];
    }
    R[m-1] = 0;
    R[m] = 0;
    for (int i = 1; i < m-1; i++){
        R[i+m] = vectorA[i+m]*omega[i-1]+vectorB[i+m]*omega[i]+vectorC[i+m]*omega[i+1]-vectorD[i+m];
    }
    R[2*m-1] = 0;
    R[2*m] = 0;
    for (int i = 1; i < m-1; i++){
        R[i+2*m] = vectorA[i+2*m]*k[i-1]+vectorB[i+2*m]*k[i]+vectorC[i+2*m]*k[i+1]-vectorD[i+2*m];
    }
    R[3*m-1] = 0;
}


void caseProp::iterativeSolver(){
    for (int i = 0; i < m; i++){
        xVelocity[i] = initXVelocity[i];
    }
    for (int i = 0; i < 10000; i++) {
        flowSolver(i);
        residualUpdate();

        double resNorm = 0;
        for (int i = 0; i < 3*m; i++){
            resNorm = resNorm + pow(R[i].value(), 2);
        }
        resNorm = sqrt(resNorm)/3.0/m;

        if ( resNorm < 1e-5 && i > 1000){
            break;
        }
        if (i==9999){
            std::cout << "did not converge: " << resNorm << std::endl;
            convergence = 0;
        }
    }
}

void caseProp::initialization(){
    iterativeSolver();
    for (int i = 0 ; i < m; i++){
        initXVelocity[i] = xVelocity[i].value();
    }
}

void caseProp::updateBoundaryVelocity(double velocity[]){
    boundaryVelocity[0] = velocity[0];
    boundaryVelocity[1] = velocity[1];
}

void caseProp::updatePressure(double inputPressure[]){
    for (int i  = 0; i < m; i++){
        pressure[i] = inputPressure[i];
    }
}
