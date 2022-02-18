import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


##flowSolver
def flowSolver(m, nu, frictionVelocity, xVelocity, yCoordinate, nut, deltaTime, gradP):    
    vectorA = np.zeros(m)
    vectorB = np.zeros(m)
    vectorC = np.zeros(m)
    vectorD = np.ones(m)
    for i in range(m):
        vectorD[i] = -vectorD[i]*gradP
    
    vectorB[0] = -1
    vectorC[0] = 0
    vectorA[m-1] = 1
    vectorB[m-1] = -1
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
def omegaSolver(m, nu, k, omega, xVelocity, yCoordinate, nut, deltaTime):
    sigma = 0.5
    alpha = 3/40
    gamma = 5/9
    #sigmaD = 0.3
    boundaryPoints = 7 ##Boundary Condition Points
    
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

    vectorA[m-1] = 1
    vectorB[m-1] = -1
    vectorD[m-1] = 0    
    for i in range(boundaryPoints, m-1):
        vectorA[i] = 2*deltaTime/(yCoordinate[i+1] - yCoordinate[i-1])*(nu+sigma*(nut[i]/2+nut[i-1]/2))/(yCoordinate[i]-yCoordinate[i-1])
        vectorB[i] = deltaTime*(-alpha*omega[i] - 2/(yCoordinate[i+1]-yCoordinate[i-1])*((nu+sigma*(nut[i]/2+nut[i-1]/2))/(yCoordinate[i]-yCoordinate[i-1])+(nu+sigma*(nut[i]/2+nut[i+1]/2))/(yCoordinate[i+1]-yCoordinate[i])))-1
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
    vectorA[m-1] = 1
    vectorB[m-1] = -1
    vectorD[0] = 0
    vectorD[m-1] = 0
    
    for i in range(1, m-1):
        vectorA[i] = 2*deltaTime/(yCoordinate[i+1] - yCoordinate[i-1])*(nu+sigmaStar*(nut[i]/2+nut[i-1]/2))/(yCoordinate[i]-yCoordinate[i-1])
        vectorB[i] = deltaTime*(-alphaStar*omega[i] - 2/(yCoordinate[i+1]-yCoordinate[i-1])*((nu+sigmaStar*(nut[i]/2+nut[i-1]/2))/(yCoordinate[i]-yCoordinate[i-1])+(nu+sigmaStar*(nut[i]/2+nut[i+1]/2))/(yCoordinate[i+1]-yCoordinate[i])))-1
        vectorC[i] = 2*deltaTime/(yCoordinate[i+1] - yCoordinate[i-1])*(nu+sigmaStar*(nut[i]/2+nut[i+1]/2))/(yCoordinate[i+1]-yCoordinate[i])
        vectorD[i] = (-nut[i]*((xVelocity[i+1]-xVelocity[i-1])/(yCoordinate[i+1]-yCoordinate[i-1]))**2)*deltaTime - k[i]     

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
        m = 100
        simpleGradingRatio = 100
        nu = 3.4e-4
        frictionVelocity = 6.37309e-2
        deltaTime = 1
    
        DNSy = np.loadtxt("180y.csv")
        DNSU = np.loadtxt("180U.csv")
        
    if reTau == 550:
        m = 100
        simpleGradingRatio = 100
        nu = 1e-4
        frictionVelocity = 5.43496e-2
        deltaTime = 100
        
        DNSy = np.loadtxt("550y.csv")
        DNSU = np.loadtxt("550U.csv")
        
    if reTau == 1000:
        m = 100
        simpleGradingRatio = 500
        nu = 5e-5
        frictionVelocity = 5.00256e-2
        deltaTime = 1
        
        DNSy = np.loadtxt("1000y.csv")
        DNSU = np.loadtxt("1000U.csv")
    
    if reTau == 2000:
        m = 100
        simpleGradingRatio = 1000
        nu = 2.3e-5
        frictionVelocity = 4.58794e-2
        deltaTime = 1
        DNSy = np.loadtxt("2000y.csv")
        DNSU = np.loadtxt("2000U.csv")
        
    if reTau == 5200:
        m = 200
        simpleGradingRatio = 2000
        nu = 8e-6
        frictionVelocity = 4.14872e-2
        deltaTime = 1
        DNSy = np.loadtxt("5200y.csv")
        DNSU = np.loadtxt("5200U.csv")
        
    ##Node Coordinate
    yCoordinate = np.zeros(m)
    xVelocity = np.zeros(m)
    
    ##Turbulence Variables
    k = 1e-8*np.ones(m)
    omega = 1e5*np.ones(m)
    nut = 1e-5*np.ones(m)
    gradP = (frictionVelocity)**2
    
    ##Node Coordinates Initilization
    gridRatio = simpleGradingRatio**(1.0/(m-2.0))
    firstNodeDist = (1.0-gridRatio)/(1.0-gridRatio**(m-1.0))
    
    tempGridSize = firstNodeDist
    for i in range(1, m):
        yCoordinate[i] = yCoordinate[i-1]+tempGridSize
        tempGridSize = tempGridSize*gridRatio

    f = interpolate.interp1d(DNSy, DNSU)
    DNSU = f(yCoordinate[0:m-1])

    for j in range(10000):
        xVelocityTemp = xVelocity - 0
        flowSolver(m, nu, frictionVelocity, xVelocity, yCoordinate, nut, deltaTime, gradP)
        omegaSolver(m, nu, k, omega, xVelocity, yCoordinate, nut, deltaTime)       
        kSolver(m, nu, k, omega, xVelocity, yCoordinate, nut, deltaTime)    
        for i in range(m):
            nut[i] = k[i]/omega[i]
        res = np.linalg.norm(xVelocity - xVelocityTemp, 1)
        if (res < 1e-8):
            print(res)
            break
    
    yPlus = yCoordinate*frictionVelocity/nu
    uPlus = xVelocity/frictionVelocity
    uPlusDNS = DNSU
    plt.semilogx(yPlus[0:m-1], uPlusDNS, "-.")
    plt.semilogx(yPlus, uPlus)
    plt.show()
    
    plt.plot(yCoordinate, uPlus)
    plt.plot(yCoordinate[0:m-1], uPlusDNS, "-.")
    plt.show()
    
    plt.plot(yCoordinate, nut)
    plt.show()
    
if __name__ == "__main__":
    mainLoop(5200)


