# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

##flowSolver
def flowSolver(m, nu, frictionVelocity, xVelocity, yCoordinate, nut, deltaTime, gradP, vectorA, vectorB, vectorC):    
    vectorB[0] = -1
    vectorC[0] = 0
    vectorA[m-1] = 0
    vectorB[m-1] = -1
    vectorD = np.ones(m)
    vectorD = -vectorD*gradP
    vectorD[0] = 0
    vectorD[m-1] = 0
    for i in range(1, m-1):
        vectorA[i] = (2/(yCoordinate[i+1] - yCoordinate[i-1])*(nu + nut[i-1]/2 + nut[i]/2)/(yCoordinate[i] - yCoordinate[i-1]))*deltaTime
        vectorB[i] = (-2/(yCoordinate[i+1] - yCoordinate[i-1])*((nu + nut[i]/2 + nut[i+1]/2)/(yCoordinate[i+1] - yCoordinate[i]) + (nu + nut[i-1]/2 + nut[i]/2)/(yCoordinate[i] - yCoordinate[i-1])))*deltaTime - 1
        vectorC[i] = (2/(yCoordinate[i+1] - yCoordinate[i-1])*(nu + nut[i]/2 + nut[i+1]/2)/(yCoordinate[i+1] - yCoordinate[i]))*deltaTime
        vectorD[i] = vectorD[i]*deltaTime - xVelocity[i]
    
    newVectorC = np.zeros(m)
    newVectorD = np.zeros(m)

    newVectorC[0] = vectorC[0]/vectorB[0]
    newVectorD[0] = vectorD[0]/vectorB[0]
    for i in range(1, m-1):
        newVectorC[i] = vectorC[i]/(vectorB[i]-vectorA[i]*newVectorC[i-1])
        newVectorD[i] = (vectorD[i]-vectorA[i]*newVectorD[i-1])/(vectorB[i]-vectorA[i]*newVectorC[i-1])
    newVectorD[m-1] = (vectorD[m-1]-vectorA[m-1]*newVectorD[m-2])/(vectorB[m-1]-vectorA[m-1]*newVectorC[m-2])
    
    xVelocity[m-1] = newVectorD[m-1]
    for i in range(m-1):
        xVelocity[m-2-i] = newVectorD[m-2-i] - newVectorC[m-2-i]*xVelocity[m-1-i]  
        
##turbulenceSolver
def omegaSolver(m, nu, k, omega, betaML, xVelocity, yCoordinate, nut, deltaTime):
    sigma = 0.5
    alpha = 3/40
    gamma = 5/9
    #sigmaD = 0.3
    boundaryPoints = 6 ##Boundary Condition Points
    
    vectorA = np.zeros(m)
    vectorB = np.zeros(m)
    vectorC = np.zeros(m)
    vectorD = np.zeros(m)
    
    vectorB[0] = -1
    vectorC[0] = 0
    vectorD[0] = -1e50
    for i in range(1, boundaryPoints):
        vectorA[i] = 0
        vectorB[i] = -1
        vectorC[i] = 0
        vectorD[i] = -6*nu/(0.00708*yCoordinate[i]**2)

    vectorA[m-1] = 0
    vectorB[m-1] = -1
    vectorD[m-1] = -1e50
    for i in range(m-2, m-1-boundaryPoints, -1):
        vectorA[i] = 0
        vectorB[i] = -1
        vectorC[i] = 0
        vectorD[i] = -6*nu/(0.00708*(2-yCoordinate[i])**2)
    
    for i in range(boundaryPoints, m-boundaryPoints):
        vectorA[i] = 2*deltaTime/(yCoordinate[i+1] - yCoordinate[i-1])*(nu+sigma*(nut[i]/2+nut[i-1]/2))/(yCoordinate[i]-yCoordinate[i-1])
        vectorB[i] = deltaTime*(-alpha*omega[i]*betaML[i] - 2/(yCoordinate[i+1]-yCoordinate[i-1])*((nu+sigma*(nut[i]/2+nut[i-1]/2))/(yCoordinate[i]-yCoordinate[i-1])+(nu+sigma*(nut[i]/2+nut[i+1]/2))/(yCoordinate[i+1]-yCoordinate[i])))-1
        vectorC[i] = 2*deltaTime/(yCoordinate[i+1] - yCoordinate[i-1])*(nu+sigma*(nut[i]/2+nut[i+1]/2))/(yCoordinate[i+1]-yCoordinate[i])
        vectorD[i] = (-gamma*((xVelocity[i+1]-xVelocity[i-1])/(yCoordinate[i+1]-yCoordinate[i-1]))**2)*deltaTime - omega[i]#-max(0, sigmaD/omega[i]*((k[i+1]-k[i-1])/(yCoordinate[i+1]-yCoordinate[i-1]))*((omega[i+1]-omega[i-1])/(yCoordinate[i+1]-yCoordinate[i-1]))))*deltaTime - omega[i]
   
    newVectorC = np.zeros(m)
    newVectorD = np.zeros(m)

    newVectorD[0] = vectorD[0]/vectorB[0]

    for i in range(1, m-1):
        newVectorC[i] = vectorC[i]/(vectorB[i]-vectorA[i]*newVectorC[i-1])
        newVectorD[i] = (vectorD[i]-vectorA[i]*newVectorD[i-1])/(vectorB[i]-vectorA[i]*newVectorC[i-1])
    newVectorD[m-1] = (vectorD[m-1]-vectorA[m-1]*newVectorD[m-2])/(vectorB[m-1]-vectorA[m-1]*newVectorC[m-2])
    
    omega[m-1] = newVectorD[m-1]
    for i in range(m-1):
        omega[m-2-i] = newVectorD[m-2-i] - newVectorC[m-2-i]*omega[m-1-i]
    
    
def kSolver(m, nu, k, omega, xVelocity, yCoordinate, nut, deltaTime):
    sigmaStar = 0.5
    alphaStar = 0.09
    
    vectorA = np.zeros(m)
    vectorB = np.zeros(m)
    vectorC = np.zeros(m)
    vectorD = np.zeros(m)
    vectorB[0] = -1
    vectorC[0] = 0
    vectorA[m-1] = 0
    vectorB[m-1] = -1
    vectorD[0] = 0
    vectorD[m-1] = 0
    
    for i in range(1, m-1):
        vectorA[i] = 2/(yCoordinate[i+1] - yCoordinate[i-1])*(nu+sigmaStar*(nut[i]/2+nut[i-1]/2))/(yCoordinate[i]-yCoordinate[i-1])
        vectorB[i] = -alphaStar*omega[i] - 2/(yCoordinate[i+1]-yCoordinate[i-1])*((nu+sigmaStar*(nut[i]/2+nut[i-1]/2))/(yCoordinate[i]-yCoordinate[i-1])+(nu+sigmaStar*(nut[i]/2+nut[i+1]/2))/(yCoordinate[i+1]-yCoordinate[i]))
        vectorC[i] = 2/(yCoordinate[i+1] - yCoordinate[i-1])*(nu+sigmaStar*(nut[i]/2+nut[i+1]/2))/(yCoordinate[i+1]-yCoordinate[i])
        vectorD[i] = -nut[i]*((xVelocity[i+1]-xVelocity[i-1])/(yCoordinate[i+1]-yCoordinate[i-1]))**2      
    
    vectorA = vectorA*deltaTime
    vectorB = vectorB*deltaTime - 1
    vectorC = vectorC*deltaTime
    vectorD = vectorD*deltaTime - k

    newVectorC = np.zeros(m)
    newVectorD = np.zeros(m)

    newVectorD[0] = vectorD[0]/vectorB[0]
    for i in range(1, m-1):
        newVectorC[i] = vectorC[i]/(vectorB[i]-vectorA[i]*newVectorC[i-1])
        newVectorD[i] = (vectorD[i]-vectorA[i]*newVectorD[i-1])/(vectorB[i]-vectorA[i]*newVectorC[i-1])
    newVectorD[m-1] = (vectorD[m-1]-vectorA[m-1]*newVectorD[m-2])/(vectorB[m-1]-vectorA[m-1]*newVectorC[m-2])
 
    k[m-1] = newVectorD[m-1]
    k[m-1] = newVectorD[m-1]
    for i in range(m-1):
        k[m-2-i] = newVectorD[m-2-i] - newVectorC[m-2-i]*k[m-1-i]

def mainLoop(reTau):
    ##Parameters
    if reTau == 180:
        n = 100
        m = 2*n-1
        simpleGradingRatio = 100
        nu = 3.4e-4
        frictionVelocity = 6.37309e-2
        deltaTime = 1
    
        DNSy = np.loadtxt("180y.csv")
        DNSU = np.loadtxt("180U.csv")
        
    if reTau == 550:
        n = 100
        m = 2*n-1
        simpleGradingRatio = 100
        nu = 1e-4
        frictionVelocity = 5.43496e-2
        deltaTime = 100
        
        DNSy = np.loadtxt("550y.csv")
        DNSU = np.loadtxt("550U.csv")
        
    if reTau == 1000:
        n = 100
        m = 2*n-1
        simpleGradingRatio = 500
        nu = 5e-5
        frictionVelocity = 5.00256e-2
        deltaTime = 1
        
        DNSy = np.loadtxt("1000y.csv")
        DNSU = np.loadtxt("1000U.csv")
    
    if reTau == 2000:
        n = 100
        m = 2*n-1
        simpleGradingRatio = 1000
        nu = 2.3e-5
        frictionVelocity = 4.58794e-2
        deltaTime = 1
        DNSy = np.loadtxt("2000y.csv")
        DNSU = np.loadtxt("2000U.csv")
        
    if reTau == 5200:
        n = 100
        m = 2*n-1
        simpleGradingRatio = 2500
        nu = 8e-6
        frictionVelocity = 4.14872e-2
        deltaTime = 1
        DNSy = np.loadtxt("5200y.csv")
        DNSU = np.loadtxt("5200U.csv")
        
    ##Node Coordinate
    yCoordinate = np.zeros(m)
    xVelocity = np.zeros(m)
    
    ##Turbulence Variables
    k = np.ones(m)
    omega = np.ones(m)
    nut = np.ones(m)
    betaML = np.ones(m)
    
    ##Node Coordinates Initilization
    gridRatio = simpleGradingRatio**(1/(n-2))
    firstNodeDist = (1-gridRatio)/(1-gridRatio**(n-1))
    
    tempGridSize = firstNodeDist
    for i in range(1, n):
        yCoordinate[i] = yCoordinate[i-1]+tempGridSize
        tempGridSize = tempGridSize*gridRatio
    for i in range(n, m):
        yCoordinate[i] = 2 - yCoordinate[m-1-i]
    
    ##Turbulence Initilizations
    nut = 1e-5*nut
    k = 1e-8*k
    omega = 1e5*omega
    gradP = (frictionVelocity)**2
    
    vectorA = np.zeros(m)
    vectorB = np.zeros(m)
    vectorC = np.zeros(m)
    
    for j in tqdm(range(10000)):
        nutTemp = nut - 0
        xVelocityTemp = xVelocity - 0
        
        flowSolver(m, nu, frictionVelocity, xVelocity, yCoordinate, nut, deltaTime, gradP, vectorA, vectorB, vectorC)
        omegaSolver(m, nu, k, omega, betaML, xVelocity, yCoordinate, nut, deltaTime)       
        kSolver(m, nu, k, omega, xVelocity, yCoordinate, nut, deltaTime)
    
        for i in range(m):
            nut[i] = k[i]/omega[i]  
    
        error = np.linalg.norm(nutTemp-nut, 1)
        error2 = np.linalg.norm(xVelocityTemp - xVelocity, 1)
    
        if error < 1e-6 and error2 < 1e-6:    
            print(j)
            break
    
    yPlus = yCoordinate[0:n-2]*frictionVelocity/nu
    uPlus = xVelocity[0:n-2]/frictionVelocity
    yPlusDNS = DNSy*frictionVelocity/nu
    uPlusDNS = DNSU
    plt.semilogx(yPlusDNS, uPlusDNS, "-.", label="observed")
    plt.semilogx(yPlus, uPlus, label="computed")
    plt.legend()
    plt.show()

    plt.plot(DNSy, DNSU, "-.", label="observed")    
    plt.plot(yCoordinate[0:n-2], uPlus, label = "computed")
    plt.legend()
    plt.show()
    
    plt.plot(yCoordinate, nut)
    plt.show()
    
    print(error2)
    print(nu*xVelocity[1]/yCoordinate[1])
    print(frictionVelocity**2)
    #print(yPlus)
    
if __name__ == "__main__":
    mainLoop(180)

