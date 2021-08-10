# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm

##Parameters
def main():
    n = 100
    m = 2*n-1
    simpleGradingRatio = 100
    nu = 1e-4
    frictionVelocity = 5.43e-2
    
    ##Node Coordinate
    yCoordinate = np.zeros(m)
    xVelocity = np.zeros(m)
    
    ##Turbulence Variables
    k = np.ones(m)
    omega = np.ones(m)
    nut = np.ones(m)
    
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
    
    Re550y = np.loadtxt("550y.csv")
    Re550U = np.loadtxt("550U.csv")
    
    for j in range(10000000):
        nutTemp = nut - 0
        xVelocityTemp = xVelocity - 0
        
        flowSolver(m, nu, frictionVelocity, yCoordinate, xVelocity, nut, gradP)
        omegaSolver(nu, m, k, omega, xVelocity, yCoordinate, nut)       
        kSolver(nu, m, k, omega, xVelocity, yCoordinate, nut)
    
        for i in range(m):
            nut[i] = k[i]/omega[i] 
    
        error = np.linalg.norm(nutTemp-nut, 1)
        error2 = np.linalg.norm(xVelocityTemp - xVelocity, 1)
        if error < 1e-6 and error2 < 1e-6:            
            break
    
        if j%10 == 0:
            yPlus = yCoordinate[0:n-2]*frictionVelocity/nu
            uPlus = xVelocity[0:n-2]/frictionVelocity
            yPlus550 = Re550y*frictionVelocity/nu
            uPlus550 = Re550U
            plt.semilogx(yPlus550, uPlus550, "-.")
            plt.semilogx(yPlus, uPlus)
            plt.show()

            
##flowSolver
def flowSolver(m, nu, frictionVelocity, yCoordinate, xVelocity, nut, gradP):    
    vectorA = np.zeros(m)
    vectorB = np.zeros(m)
    vectorC = np.zeros(m)
    vectorD = np.zeros(m)
    
    vectorB[0] = -1
    vectorC[0] = 0
    vectorA[m-1] = 0
    vectorB[m-1] = -1
    vectorD = np.ones(m)
    vectorD = -vectorD*gradP
    vectorD[0] = 0
    vectorD[m-1] = 0
    for i in range(1, m-1):
        vectorA[i] = 2/(yCoordinate[i+1] - yCoordinate[i-1])*(nu + nut[i-1]/2 + nut[i]/2)/(yCoordinate[i] - yCoordinate[i-1])
        vectorB[i] = -2/(yCoordinate[i+1] - yCoordinate[i-1])*((nu + nut[i]/2 + nut[i+1]/2)/(yCoordinate[i+1] - yCoordinate[i]) + (nu + nut[i-1]/2 + nut[i]/2)/(yCoordinate[i] - yCoordinate[i-1]))
        vectorC[i] = 2/(yCoordinate[i+1] - yCoordinate[i-1])*(nu + nut[i]/2 + nut[i+1]/2)/(yCoordinate[i+1] - yCoordinate[i])
        vectorD[i] = vectorD[i]
    
    newVectorC = np.zeros(m)
    newVectorD = np.zeros(m)

    newVectorC[0] = vectorC[0]/vectorB[0]
    newVectorD[0] = vectorD[0]/vectorB[0]
    for i in range(1, m-1):
        newVectorC[i] = vectorC[i]/(vectorB[i]-vectorA[i]*newVectorC[i-1])
        newVectorD[i] = (vectorD[i]-vectorA[i]*newVectorD[i-1])/(vectorB[i]-vectorA[i]*newVectorC[i-1])
    newVectorD[m-1] = (vectorD[m-1]-vectorA[m-1]*newVectorD[m-2])/(vectorB[m-1]-vectorA[m-1]*newVectorC[m-2])
    
    xVelocityTemp = xVelocity - 0
    xVelocity[m-1] = newVectorD[m-1]
    for i in range(m-1):
        xVelocity[m-2-i] = newVectorD[m-2-i] - newVectorC[m-2-i]*xVelocity[m-1-i]  
    xVelocity = 0.45*xVelocity + 0.55*xVelocityTemp
    
##turbulenceSolver
def omegaSolver(nu, m, k, omega, xVelocity, yCoordinate, nut):
    sigma = 0.5
    alpha = 3/40
    gamma = 5/9
    boundaryPoints = 7 ##Boundary Condition Points
    
    vectorA = np.zeros(m)
    vectorB = np.zeros(m)
    vectorC = np.zeros(m)
    vectorD = np.zeros(m)
    
    vectorB[0] = -1
    vectorC[0] = 0
    vectorD[0] = -1e10
    for i in range(1, boundaryPoints):
        vectorA[i] = 0
        vectorB[i] = -1
        vectorC[i] = 0
        vectorD[i] = -6*nu/(0.00708*yCoordinate[i]**2)

    vectorA[m-1] = 0
    vectorB[m-1] = -1
    vectorD[m-1] = -1e10
    for i in range(m-2, m-1-boundaryPoints, -1):
        vectorA[i] = 0
        vectorB[i] = -1
        vectorC[i] = 0
        vectorD[i] = -6*nu/(0.00708*(2-yCoordinate[i])**2)
    
    for i in range(boundaryPoints, m-boundaryPoints):
        vectorA[i] = 2/(yCoordinate[i+1] - yCoordinate[i-1])*(nu+sigma*(nut[i]/2+nut[i-1]/2))/(yCoordinate[i]-yCoordinate[i-1])
        vectorB[i] = -alpha*omega[i] - 2/(yCoordinate[i+1]-yCoordinate[i-1])*((nu+sigma*(nut[i]/2+nut[i-1]/2))/(yCoordinate[i]-yCoordinate[i-1])+(nu+sigma*(nut[i]/2+nut[i+1]/2))/(yCoordinate[i+1]-yCoordinate[i]))
        vectorC[i] = 2/(yCoordinate[i+1] - yCoordinate[i-1])*(nu+sigma*(nut[i]/2+nut[i+1]/2))/(yCoordinate[i+1]-yCoordinate[i])
        vectorD[i] = -gamma*((xVelocity[i+1]-xVelocity[i-1])/(yCoordinate[i+1]-yCoordinate[i-1]))**2
               
    newVectorC = np.zeros(m)
    newVectorD = np.zeros(m)

    newVectorD[0] = vectorD[0]/vectorB[0]

    for i in range(1, m-1):
        newVectorC[i] = vectorC[i]/(vectorB[i]-vectorA[i]*newVectorC[i-1])
        newVectorD[i] = (vectorD[i]-vectorA[i]*newVectorD[i-1])/(vectorB[i]-vectorA[i]*newVectorC[i-1])
    newVectorD[m-1] = (vectorD[m-1]-vectorA[m-1]*newVectorD[m-2])/(vectorB[m-1]-vectorA[m-1]*newVectorC[m-2])
    
    omegaTemp = np.zeros(m)
    omegaTemp = omega - 0
    omega[m-1] = newVectorD[m-1]
    for i in range(m-1):
        omega[m-2-i] = newVectorD[m-2-i] - newVectorC[m-2-i]*omega[m-1-i]
    omega = 0.55*omegaTemp + 0.45*omega
    error = np.linalg.norm(omegaTemp-omega, 1)
    return error
    
    
def kSolver(nu, m, k, omega, xVelocity, yCoordinate, nut):
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

    newVectorC = np.zeros(m)
    newVectorD = np.zeros(m)

    newVectorD[0] = vectorD[0]/vectorB[0]
    for i in range(1, m-1):
        newVectorC[i] = vectorC[i]/(vectorB[i]-vectorA[i]*newVectorC[i-1])
        newVectorD[i] = (vectorD[i]-vectorA[i]*newVectorD[i-1])/(vectorB[i]-vectorA[i]*newVectorC[i-1])
    newVectorD[m-1] = (vectorD[m-1]-vectorA[m-1]*newVectorD[m-2])/(vectorB[m-1]-vectorA[m-1]*newVectorC[m-2])
 
    kTemp = np.zeros(m)
    kTemp = k - 0
    k[m-1] = newVectorD[m-1]
    k[m-1] = newVectorD[m-1]
    for i in range(m-1):
        k[m-2-i] = newVectorD[m-2-i] - newVectorC[m-2-i]*k[m-1-i]
    k = 0.45*k + 0.55*kTemp
    error = np.linalg.norm(kTemp-k, 1)
    return error

if __name__ == "__main__":
    main()
