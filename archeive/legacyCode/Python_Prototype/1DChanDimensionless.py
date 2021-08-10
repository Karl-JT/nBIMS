# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:21:55 2019

@author: yjuntao
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    nodeNumber = 100
    reTau = 543.0
    yCoordinate = simpleGradingGridGen(nodeNumber)
    betaML = np.ones(nodeNumber)
    yPlus = reTau*(1.0-yCoordinate)
    cMatrix, aMatrix = secondOrderFiniteDiffDiscret(yCoordinate, nodeNumber)
    k, omega, nu = initialization(nodeNumber, yPlus)
    u = np.zeros(nodeNumber)
    relaxFactor = 0.01
    blendFactor = 0.8
    u = np.zeros(nodeNumber)

    for i in range(2):
        dudy = -reTau*yCoordinate/(1+nu)        
        dnudy = dnudyUpdate(nu, nodeNumber, cMatrix)
        dkdy = dkdyUpdate(k, nodeNumber, cMatrix)
        dody = dodyUpdate(omega, nodeNumber, cMatrix)
        omegaSource = omegaSourceUpdate(k, omega, dudy, reTau)
        tA, tB, tC, tD = omegaSystemEqn(nodeNumber, dnudy, nu, cMatrix, aMatrix, omegaSource, omega, relaxFactor, blendFactor, yPlus)
        omega = thomasSolver(tA, tB, tC, tD, relaxFactor, omega)
        kSource = kSourceUpdate(nu, dudy, reTau, k, omega, dkdy, dody)
        tA, tB, tC, tD = kSystemEqn(nodeNumber, dnudy, nu, cMatrix, aMatrix, kSource, k, relaxFactor, blendFactor)
        k = thomasSolver(tA, tB, tC, tD, relaxFactor, k)
        nu = nuUpdate(k, omega)
        xVelocity = u
    
        if i % 1000 == 0: 
            plt.plot(yCoordinate, k)
            plt.show()
            plt.plot(yCoordinate[0:88], omega[0:88])
            plt.show()
 #           plt.plot(yCoordinate, nu)
 #           plt.show()
            for j in range(nodeNumber-2, -1, -1):
                u[j] = u[j+1] + 1/2*(yCoordinate[j]-yCoordinate[j+1])*(dudy[j]+dudy  [j+1])
            error = np.linalg.norm(xVelocity-u)
            plt.plot(yCoordinate, u)
            plt.show()
            print(error)
            
def simpleGradingGridGen(nodeNumber):
    yCoordinate = np.zeros(nodeNumber)
    simpleGradingRatio = 100
    
    gridRatio = simpleGradingRatio**(1/(nodeNumber-2))
    firstNodeDist = (1-gridRatio)/(1-gridRatio**(nodeNumber-1))*simpleGradingRatio
    
    tempGridSize = firstNodeDist
    for i in range(1, nodeNumber-1):
        yCoordinate[i] = yCoordinate[i-1]+tempGridSize
        tempGridSize = tempGridSize/gridRatio
    yCoordinate[nodeNumber-1] = 1.0
    return yCoordinate

def secondOrderFiniteDiffDiscret(yCoordinate, nodeNumber):
    cMatrix = np.zeros((nodeNumber, 5))
    aMatrix = np.zeros((nodeNumber, 7))
    
    ya = yCoordinate[1]-yCoordinate[0]
    yb = yCoordinate[2]-yCoordinate[0]
    yc = yCoordinate[3]-yCoordinate[0]
    
    cMatrix[0,0]=0
    cMatrix[0,1]=0
    cMatrix[0,2]=(ya+yb)/(ya*yb)
    cMatrix[0,3]=(-yb)/ya/(ya-yb)
    cMatrix[0,4]=ya/yb/(ya-yb)
    
    aMatrix[0,0]=0
    aMatrix[0,1]=0
    aMatrix[0,2]=0
    aMatrix[0,3]=2*(ya+yb+yc)/ya/yb/yc
    aMatrix[0,4]=2*(yb+yc)/ya/(ya-yb)/(yc-ya)
    aMatrix[0,5]=2*(yc+ya)/yb/(yb-yc)/(ya-yb)
    aMatrix[0,6]=2*(ya+yb)/yc/(yb-yc)/(yc-ya)

    ya = yCoordinate[0]-yCoordinate[1]
    yb = yCoordinate[2]-yCoordinate[1]
    yc = yCoordinate[3]-yCoordinate[1]
    
    cMatrix[1,0]=0
    cMatrix[1,1]=-yb/ya/(ya-yb)
    cMatrix[1,2]=-(ya+yb)/(ya*yb)
    cMatrix[1,3]=ya/yb/(ya-yb)
    cMatrix[1,4]=0
    
    aMatrix[1,0]=0
    aMatrix[1,1]=0
    aMatrix[1,2]=2*(yb+yc)/ya/(ya-yb)/(yc-ya)
    aMatrix[1,3]=2*(ya+yb+yc)/ya/yb/yc
    aMatrix[1,4]=2*(ya+yc)/yb/(yb-yc)/(ya-yb)
    aMatrix[1,5]=2*(yb+ya)/yc/(yb-yc)/(yc-ya)
    aMatrix[1,6]=0
    
    for i in range(2, nodeNumber-1):
        ya=yCoordinate[i-1]-yCoordinate[i]
        yb=yCoordinate[i+1]-yCoordinate[i]
        yc=yCoordinate[i-2]-yCoordinate[i]
        
        cMatrix[i,0]=0
        cMatrix[i,1]=-yb/ya/(ya-yb)
        cMatrix[i,2]=-(ya+yb)/(ya*yb)
        cMatrix[i,3]=ya/yb/(ya-yb)
        cMatrix[i,4]=0
        
        aMatrix[i,0]=0
        aMatrix[i,1]=2*(ya+yb)/yc/(yb-yc)/(yc-ya)
        aMatrix[i,2]=2*(yb+yc)/ya/(ya-yb)/(yc-ya)
        aMatrix[i,3]=2*(ya+yb+yc)/ya/yb/yc
        aMatrix[i,4]=2*(ya+yc)/yb/(yb-yc)/(ya-yb)
        aMatrix[i,5]=0
        aMatrix[i,6]=0
        
    ya=yCoordinate[nodeNumber-2]-yCoordinate[nodeNumber-1]
    yb=yCoordinate[nodeNumber-3]-yCoordinate[nodeNumber-1]
    yc=yCoordinate[nodeNumber-4]-yCoordinate[nodeNumber-1]
    
    cMatrix[nodeNumber-1,0]=ya/yb/(ya-yb)
    cMatrix[nodeNumber-1,1]=-yb/ya/(ya-yb)
    cMatrix[nodeNumber-1,2]=-(ya+yb)/(ya*yb)
    cMatrix[nodeNumber-1,3]=0
    cMatrix[nodeNumber-1,4]=0
    
    aMatrix[nodeNumber-1,0]=2*(ya+yb)/yc/(yb-yc)/(yc-ya)
    aMatrix[nodeNumber-1,1]=2*(ya+yc)/yb/(yb-yc)/(ya-yb)
    aMatrix[nodeNumber-1,2]=2*(yb+yc)/ya/(ya-yb)/(yc-ya)
    aMatrix[nodeNumber-1,3]=2*(ya+yb+yc)/ya/yb/yc
    aMatrix[nodeNumber-1,4]=0
    aMatrix[nodeNumber-1,5]=0
    aMatrix[nodeNumber-1,6]=0
    
    return cMatrix, aMatrix

def initialization(nodeNumber, yPlus):
    k=np.array([-0.00000000e+00,  6.75790405e-12,  5.67069530e-10,  1.33116327e-08,
        1.59487053e-07,  1.24964179e-06,  4.94868429e-06,  1.37948185e-05,
        3.12845769e-05,  6.20179965e-05,  1.11833911e-04,  1.87856358e-04,
        2.98328460e-04,  4.52063289e-04,  6.57348317e-04,  9.20286118e-04,
        1.24288379e-03,  1.62158436e-03,  2.04702410e-03,  2.50534310e-03,
        2.98060065e-03,  3.45734887e-03,  3.92253512e-03,  4.36641025e-03,
        4.78258107e-03,  5.16754455e-03,  5.52002084e-03,  5.84029220e-03,
        6.12964710e-03,  6.38995716e-03,  6.62337712e-03,  6.83214420e-03,
        7.01845160e-03,  7.18437416e-03,  7.33182924e-03,  7.46256052e-03,
        7.57813607e-03,  7.67995484e-03,  7.76925766e-03,  7.84714051e-03,
        7.91456829e-03,  7.97238834e-03,  8.02134319e-03,  8.06208220e-03,
        8.09517209e-03,  8.12110634e-03,  8.14031345e-03,  8.15316430e-03,
        8.15997849e-03,  8.16102993e-03,  8.15655173e-03,  8.14674039e-03,
        8.13175951e-03,  8.11174301e-03,  8.08679786e-03,  8.05700656e-03,
        8.02242925e-03,  7.98310552e-03,  7.93905608e-03,  7.89028413e-03,
        7.83677667e-03,  7.77850559e-03,  7.71542868e-03,  7.64749058e-03,
        7.57462365e-03,  7.49674876e-03,  7.41377618e-03,  7.32560637e-03,
        7.23213085e-03,  7.13323325e-03,  7.02879034e-03,  6.91867330e-03,
        6.80274928e-03,  6.68088312e-03,  6.55293959e-03,  6.41878608e-03,
        6.27829588e-03,  6.13135232e-03,  5.97785389e-03,  5.81772064e-03,
        5.65090220e-03,  5.47738785e-03,  5.29721916e-03,  5.11050598e-03,
        4.91744657e-03,  4.71835315e-03,  4.51368426e-03,  4.30408585e-03,
        4.09044362e-03,  3.87394943e-03,  3.65618561e-03,  3.43923109e-03,
        3.22579317e-03,  3.01936673e-03,  2.82441629e-03,  2.64656246e-03,
        2.49272656e-03,  2.37114426e-03,  2.29111452e-03,  2.26235290e-03])
    k = np.flip(k, 0)/(0.0543**2)
    omega=np.array([1.00000000e+10, 3.94530032e+05, 9.40528981e+04, 3.98457503e+04,
       2.13568504e+04, 6.77120301e+03, 3.02809764e+03, 1.63974420e+03,
       9.99782533e+02, 6.60633265e+02, 4.62940159e+02, 3.39590670e+02,
       2.58646382e+02, 2.03418304e+02, 1.64511889e+02, 1.36310345e+02,
       1.15293403e+02, 9.91812993e+01, 8.64715626e+01, 7.61692767e+01,
       6.76165335e+01, 6.03776796e+01, 5.41607306e+01, 4.87647099e+01,
       4.40457574e+01, 3.98961875e+01, 3.62319264e+01, 3.29850596e+01,
       3.00993641e+01, 2.75275304e+01, 2.52293387e+01, 2.31703769e+01,
       2.13210822e+01, 1.96559859e+01, 1.81530992e+01, 1.67934007e+01,
       1.55604051e+01, 1.44397982e+01, 1.34191256e+01, 1.24875285e+01,
       1.16355188e+01, 1.08547875e+01, 1.01380414e+01, 9.47886370e+00,
       8.87159612e+00, 8.31123705e+00, 7.79335520e+00, 7.31401550e+00,
       6.86971556e+00, 6.45733109e+00, 6.07406897e+00, 5.71742673e+00,
       5.38515758e+00, 5.07524012e+00, 4.78585208e+00, 4.51534747e+00,
       4.26223664e+00, 4.02516900e+00, 3.80291770e+00, 3.59436642e+00,
       3.39849756e+00, 3.21438202e+00, 3.04117005e+00, 2.87808325e+00,
       2.72440745e+00, 2.57948642e+00, 2.44271624e+00, 2.31354035e+00,
       2.19144510e+00, 2.07595578e+00, 1.96663309e+00, 1.86307004e+00,
       1.76488905e+00, 1.67173955e+00, 1.58329570e+00, 1.49925446e+00,
       1.41933390e+00, 1.34327172e+00, 1.27082405e+00, 1.20176451e+00,
       1.13588347e+00, 1.07298769e+00, 1.01290029e+00, 9.55461110e-01,
       9.00527591e-01, 8.47976340e-01, 7.97705476e-01, 7.49638074e-01,
       7.03727003e-01, 6.59961621e-01, 6.18376866e-01, 5.79065474e-01,
       5.42194082e-01, 5.08023914e-01, 4.76936160e-01, 4.49460614e-01,
       4.26302841e-01, 4.08359771e-01, 3.96707911e-01, 3.92549133e-01])
    omega = np.flip(omega, 0)*1e-4/(0.0543**2)
    nu = np.array([-0.00000000e+00,  1.71289978e-17,  6.02926163e-15,  3.34079108e-13,
        7.46772348e-12,  1.84552404e-10,  1.63425519e-09,  8.41278691e-09,
        3.12913817e-08,  9.38765875e-08,  2.41573147e-07,  5.53184687e-07,
        1.15342213e-06,  2.22233339e-06,  3.99574961e-06,  6.75140333e-06,
        1.07801814e-05,  1.63496987e-05,  2.36728011e-05,  3.28917801e-05,
        4.40809443e-05,  5.72620361e-05,  7.24239697e-05,  8.95403716e-05,
        1.08582105e-04,  1.29524771e-04,  1.52352397e-04,  1.77058713e-04,
        2.03647063e-04,  2.32129692e-04,  2.62526783e-04,  2.94865475e-04,
        3.29178957e-04,  3.65505662e-04,  4.03888567e-04,  4.44374588e-04,
        4.87014060e-04,  5.31860260e-04,  5.78968997e-04,  6.28398208e-04,
        6.80207597e-04,  7.34458257e-04,  7.91212317e-04,  8.50532560e-04,
        9.12482036e-04,  9.77123656e-04,  1.04451975e-03,  1.11473161e-03,
        1.18781897e-03,  1.26383947e-03,  1.34284806e-03,  1.42489633e-03,
        1.51003186e-03,  1.59829738e-03,  1.68973000e-03,  1.78436026e-03,
        1.88221113e-03,  1.98329698e-03,  2.08762237e-03,  2.19518080e-03,
        2.30595330e-03,  2.41990701e-03,  2.53699351e-03,  2.65714711e-03,
        2.78028298e-03,  2.90629511e-03,  3.03505420e-03,  3.16640527e-03,
        3.30016519e-03,  3.43612004e-03,  3.57402220e-03,  3.71358734e-03,
        3.85449118e-03,  3.99636601e-03,  4.13879706e-03,  4.28131865e-03,
        4.42341009e-03,  4.56449149e-03,  4.70391938e-03,  4.84098223e-03,
        4.97489606e-03,  5.10480025e-03,  5.22975379e-03,  5.34873259e-03,
        5.46062843e-03,  5.56425094e-03,  5.65833430e-03,  5.74155183e-03,
        5.81254321e-03,  5.86996168e-03,  5.91255238e-03,  5.93927844e-03,
        5.94951748e-03,  5.94335551e-03,  5.92200073e-03,  5.88830785e-03,
        5.84731398e-03,  5.80650796e-03,  5.77531845e-03,  5.76323500e-03])
    nu = np.flip(nu, 0)/1e-4
    return k, omega, nu   

def dnudyUpdate(nu, nodeNumber, cMatrix):
    dnudy = np.zeros(nodeNumber)
    dnudy[0] = cMatrix[0,0]*nu[0] + cMatrix[0,1]*nu[1] + cMatrix[0,2]*nu[2]
    for i in range(1, nodeNumber-1):
        dnudy[i]=cMatrix[i,1]*nu[i-1]+cMatrix[i,2]*nu[i]+cMatrix[i,3]*nu[i+1]
    dnudy[nodeNumber-1]=cMatrix[nodeNumber-1,0]*nu[nodeNumber-3]+cMatrix[nodeNumber-1,1]*nu[nodeNumber-2]+cMatrix[nodeNumber-1,2]*nu[nodeNumber-1] 
    return dnudy

def dkdyUpdate(k, nodeNumber, cMatrix):
    dkdy = np.zeros(nodeNumber)
    for i in range(1, nodeNumber-1):
        dkdy[i]=cMatrix[i,1]*k[i-1]+cMatrix[i,2]*k[i]+cMatrix[i,3]*k[i+1]
    dkdy[nodeNumber-1]=cMatrix[nodeNumber-1,0]*k[nodeNumber-3]+cMatrix[nodeNumber-1,1]*k[nodeNumber-2]+cMatrix[nodeNumber-1,2]*k[nodeNumber-1]     
    #dkdy[nodeNumber-1]=0 Questionable Choice
    return dkdy

def dodyUpdate(omega, nodeNumber, cMatrix):
    dody = np.zeros(nodeNumber)
    for i in range(1, nodeNumber-1):
        dody[i]=cMatrix[i,1]*omega[i-1]+cMatrix[i,2]*omega[i]+cMatrix[i,3]*omega[i+1]
    dody[nodeNumber-1]=cMatrix[nodeNumber-1,0]*omega[nodeNumber-3]+cMatrix[nodeNumber-1,1]*omega[nodeNumber-2]+cMatrix[nodeNumber-1,2]*omega[nodeNumber-1] 
    return dody

def kSystemEqn(nodeNumber, dnudy, nu, cMatrix, aMatrix, S, x, relaxFactor, blendFactor):
    
    tA = np.zeros(nodeNumber)
    tB = np.zeros(nodeNumber)
    tC = np.zeros(nodeNumber)
    tD = np.zeros(nodeNumber)

    tA[0] = cMatrix[0, 2]
    tB[0] = cMatrix[0, 3]
    tC[0] = cMatrix[0, 4]
    tD[0] = 0
    
    tA[1] = -(dnudy[1]/2)*cMatrix[1, 1] - (1+nu[1]/2)*aMatrix[1, 2]
    tB[1] = -(dnudy[1]/2)*cMatrix[1, 2] - (1+nu[1]/2)*aMatrix[1, 3]
    tC[1] = -(dnudy[1]/2)*cMatrix[1, 3] - (1+nu[1]/2)*aMatrix[1, 4]
    tD[1] = S[1]+(1+nu[1]/2)*aMatrix[1, 5]*x[3]
    
    for i in range(2, nodeNumber-2):
        tA[i] = -(dnudy[i]/2)*cMatrix[i, 1] - (1+nu[i]/2)*aMatrix[i, 2]
        tB[i] = -(dnudy[i]/2)*cMatrix[i, 2] - (1+nu[i]/2)*aMatrix[i, 3]
        tC[i] = -(dnudy[i]/2)*cMatrix[i, 3] - (1+nu[i]/2)*aMatrix[i, 4]
        tD[i] = S[i]+(1+nu[i]/2)*aMatrix[i, 1]*x[i-3]
    
    tA[nodeNumber-2] = cMatrix[nodeNumber-2, 0]
    tB[nodeNumber-2] = cMatrix[nodeNumber-2, 1]
    tC[nodeNumber-2] = cMatrix[nodeNumber-2, 2]
    tD[nodeNumber-2] = 0    

    tA[nodeNumber-1] = 0
    tB[nodeNumber-1] = 1
    tC[nodeNumber-1] = 0
    tD[nodeNumber-1] = 0
    
    tA[1:nodeNumber-1] = tA[1:nodeNumber-1]*blendFactor
    tC[1:nodeNumber-1] = tC[1:nodeNumber-1]*blendFactor
    for i in range(1, nodeNumber-1):
        tD[i] = tD[i] - (1-blendFactor)*(tA[i]*x[i-1]+tC[i]*x[i+1])
    return tA, tB, tC, tD

def omegaSystemEqn(nodeNumber, dnudy, nu, cMatrix, aMatrix, S, x, relaxFactor, blendFactor, yPlus):
    tA = np.zeros(nodeNumber)
    tB = np.zeros(nodeNumber)
    tC = np.zeros(nodeNumber)
    tD = np.zeros(nodeNumber)
    
    tA[0] = cMatrix[0, 2]
    tB[0] = cMatrix[0, 3]
    tC[0] = cMatrix[0, 4]
    tD[0] = 0
    
    tA[1] = -(dnudy[1]/2)*cMatrix[1, 1] - (1+nu[1]/2)*aMatrix[1, 2]
    tB[1] = -(dnudy[1]/2)*cMatrix[1, 2] - (1+nu[1]/2)*aMatrix[1, 3]
    tC[1] = -(dnudy[1]/2)*cMatrix[1, 3] - (1+nu[1]/2)*aMatrix[1, 4]
    tD[1] = S[1]+(1+nu[1]/2)*aMatrix[1, 5]*x[3]
    
    for i in range(2, nodeNumber-7):
        tA[i] = -(dnudy[i]/2)*cMatrix[i, 1] - (1+nu[i]/2)*aMatrix[i, 2]
        tB[i] = -(dnudy[i]/2)*cMatrix[i, 2] - (1+nu[i]/2)*aMatrix[i, 3]
        tC[i] = -(dnudy[i]/2)*cMatrix[i, 3] - (1+nu[i]/2)*aMatrix[i, 4]
        tD[i] = S[i]+(1+nu[i]/2)*aMatrix[i,1]*x[i-2]
    
    for i in range(nodeNumber-7, nodeNumber-1):
        tA[i] = 0
        tB[i] = 1
        tC[i] = 0
        tD[i] = -6/0.0072/yPlus[i]**2
    tA[nodeNumber-1] = 0
    tB[nodeNumber-1] = 1
    tC[nodeNumber-1] = 0
    tD[nodeNumber-1] = -1e10
    
    print(tA)
    print(tB)
    print(tC)
    print(tD)

    tA[1:nodeNumber-7] = tA[1:nodeNumber-7]*blendFactor
    tC[1:nodeNumber-7] = tC[1:nodeNumber-7]*blendFactor
    for i in range(1, nodeNumber-7):
        tD[i] = tD[i] - (1-blendFactor)*(tA[i]*x[i-1]+tC[i]*x[i+1])
    return tA, tB, tC, tD

def kSourceUpdate(nu, dudy, reTau, k, omega, dkdy, dody):
    kSource = nu*dudy**2-0.09*reTau**2*k*omega
    return kSource

def thomasSolver(tA, tB, tC, tD, relaxFactor, x):  
    
    newVectorC = np.zeros(tC.size)
    newVectorD = np.zeros(tD.size)
    tempField = x - 0

    newVectorC[0] = tC[0]/tB[0]
    newVectorD[0] = tD[0]/tB[0]
    
    for i in range(1, tB.size-1):
        newVectorC[i] = tC[i]/(tB[i]-tA[i]*newVectorC[i-1])
        newVectorD[i] = (tD[i]-tA[i]*newVectorD[i-1])/(tB[i]-tA[i]*newVectorC[i-1])
    newVectorD[tD.size-1] = (tD[tD.size-1]-tA[tA.size-1]*newVectorD[tD.size-2])/(tB[tB.size-1]-tA[tB.size-1]*newVectorC[tC.size-2])
    
    x[tD.size-1] = newVectorD[tD.size-1]
    for i in range(tB.size-1):
        x[tB.size-2-i] = newVectorD[tB.size-2-i] - newVectorC[tB.size-2-i]*x[tB.size-1-i]
    x = (1-relaxFactor)*tempField + relaxFactor*x
    return x

def omegaSourceUpdate(k, omega, dudy, reTau):
    omegaSource = 0.52*dudy**2-0.072*reTau**2*omega**2
    return omegaSource

def nuUpdate(k, omega):
    nu = k/omega
    return nu

if __name__ == "__main__":
    main()

