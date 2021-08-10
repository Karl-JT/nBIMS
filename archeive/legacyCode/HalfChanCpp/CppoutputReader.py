# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:48:22 2019

@author: yjuntao
"""

import numpy as np
import matplotlib.pyplot as plt

yCoordinate = np.loadtxt("yFile.csv")
xVelocity = np.loadtxt("uFile.csv")
betaML = np.loadtxt("betaFile.csv")

yDNS = np.loadtxt("../5200y.csv")
uDNS = np.loadtxt("../5200U.csv")

##Re=5200
frictionVelocity = 4.14872e-2
nu = 8e-6

# =============================================================================
# ##Re=550
# frictionVelocity = 5.43496e-2
# nu = 1e-4
# =============================================================================

# =============================================================================
# ##Re=180
# frictionVelocity = 6.37309e-2
# nu = 3.5e-4
# =============================================================================

# =============================================================================
# ##Re=2000
# frictionVelocity = 4.58794e-2
# nu = 2.3e-5
# =============================================================================

# =============================================================================
# ##Re=1000
# frictionVelocity = 5.00256e-2
# nu = 5e-5
# =============================================================================



plt.semilogx(yDNS*frictionVelocity/nu, uDNS*frictionVelocity, label = "DNS")
plt.semilogx(yCoordinate*frictionVelocity/nu, xVelocity, label = "Optimized RANS")
plt.title("Re_tau = 5200")
plt.xlabel("yPlus")
plt.ylabel("U+")
plt.show()

plt.semilogx(yCoordinate*frictionVelocity/nu, betaML)
plt.title("Re_tau = 5200")
plt.xlabel("yPlus")
plt.ylabel("Optimal beta function")
plt.show()

#plt.plot(yDNS, uDNS*6.37309e-2)
#plt.plot(yCoordinate, xVelocity)
#plt.plot(yCoordinate, dnsVelocity)
#plt.show()