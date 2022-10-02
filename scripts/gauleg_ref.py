import numpy as np
from NSE2d import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# print(rank, size)

#Gaussian Prior (took from m = 1.3)  quadrature reference results =  -1.7729268 with 256 points
Obs_True = np.array([0.73055473, 0.72695932, 0.26300638, 0.11710155, 1.22372166, -0.0090397, 0.25021389, -0.12153667, 0.92661027, 0.31401691, 1.48933976, -0.04004448, 1.05130838, -0.69301329, 2.02790285, -0.83326638, 1.03951154, -0.91920342, 1.75492334, -0.49436064]) #gaussian prior
#Uniform Prior (took from m = 0.8) quadrature reference results = -0.997903225834290 with 64 points noise = 0.01, -0.12910684141100986 with noise = 1.0
Obs_True = np.array([0.63979175, 0.79484298, 0.17199418, 0.19541597, 1.13408598, 0.07899129, 0.17270835, -0.02744946, 0.84578063, 0.41139245, 1.11225036, 0.32293072, 0.65912191, -0.30282125, 1.62780146, -0.42925516, 0.65568777, -0.5237915, 1.35274024, -0.08863465]) #unform prior
Noise_Var = 1.0
n = 64
x, w = np.polynomial.legendre.leggauss(n)
# x, w = np.polynomial.hermite.hermgauss(n)

num_sum = 0
dem_sum = 0

# if rank == 0:
#     print(int(n/size))
for i in range(int(n/size)):
    if rank == 0:
        print(i)
    qoi, obs = solve_NS(x[rank+i*size])
    obs = np.array(obs).flatten()
    num = qoi*w[rank+i*size]*np.exp(-0.5*np.square(np.subtract(Obs_True, obs))/Noise_Var)
    dem = w[rank+i*size]*np.exp(-0.5*np.square(np.subtract(Obs_True, obs))/Noise_Var)
    num_sum = num_sum+np.sum(num)
    dem_sum = dem_sum+np.sum(dem)
    print("rank", rank, "w", w[rank+i*size], "x", x[rank+i*size], "qoi", qoi, "num_sum, dem_sum", np.sum(num), np.sum(dem))
    sys.stdout.flush()

num_sum = comm.allreduce(num_sum, op=MPI.SUM)
dem_sum = comm.allreduce(dem_sum, op=MPI.SUM)

ref = num_sum/dem_sum

print(ref)
