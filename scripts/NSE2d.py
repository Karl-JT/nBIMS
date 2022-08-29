from re import M
import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.integrate import newton_cotes

def init_mesh(N):
    global  x, y, dx, X, Y, AliasCor, Kx, Ky, K2, K2inv, vx, vy, Vx_hat, Vy_hat
    
    x = np.linspace(0, 1, N+1, endpoint=False)
    y = np.linspace(0, 1, N+1, endpoint=False)

    dx = 1.0/(N+1)

    X, Y = np.meshgrid(x, y)

    kx = np.append(np.arange(0, N/2+1), np.arange(-N/2, 0))*math.pi/0.5
    ky = np.append(np.arange(0, N/2+1), np.arange(-N/2, 0))*math.pi/0.5

    AliasCor = np.ones((N+1, N+1))
    jx = np.arange(N//4+1, N//4*3+1) 
    jy = np.arange(N//4+1, N//4*3+1)
    AliasCor[jx, :] = 0
    AliasCor[:, jy] = 0

    Kx, Ky = np.meshgrid(kx, ky)
    K2 = np.square(Kx)+np.square(Ky)

    K2inv = np.zeros_like(K2)
    K2inv[K2 > 1e-12] = 1./K2[K2 > 1e-12]

    vx = np.zeros_like(X)
    vy = np.zeros_like(Y)
    Vx_hat = np.fft.fft2(vx)
    Vy_hat = np.fft.fft2(vy)

def rf1(m, t):
    f = -m*np.multiply(np.cos(2*math.pi*X), np.sin(2*math.pi*Y))*(math.exp(t)-1.0)
    return f

def rf2(m, t):
    f = m*np.multiply(np.sin(2*math.pi*X), np.cos(2*math.pi*Y))*(math.exp(t)-1.0)
    return f

def rhs(m, t):
    global vx, vy

    f1 = rf1(m, t)
    f2 = rf2(m, t)

    F1_hat = np.fft.fft2(f1)
    F2_hat = np.fft.fft2(f2)

    vx = np.fft.ifft2(Vx_hat)
    vy = np.fft.ifft2(Vy_hat)

    vxx = np.fft.ifft2(1j*np.multiply(Kx, Vx_hat))
    vxy = np.fft.ifft2(1j*np.multiply(Ky, Vx_hat))
    vyx = np.fft.ifft2(1j*np.multiply(Kx, Vy_hat))
    vyy = np.fft.ifft2(1j*np.multiply(Ky, Vy_hat))

    Vvx = np.multiply(vx, vxx) + np.multiply(vy, vxy)
    Vvy = np.multiply(vx, vyx) + np.multiply(vy, vyy)

    Vvx_hat = np.multiply(AliasCor, np.fft.fft2(Vvx))
    Vvy_hat = np.multiply(AliasCor, np.fft.fft2(Vvy))

    I1 = np.multiply(np.multiply(Kx, F1_hat)+np.multiply(Ky, F2_hat), K2inv)
    I2 = np.multiply(np.multiply(Kx, Vvx_hat)+np.multiply(Ky, Vvy_hat), K2inv)

    Hx = F1_hat - np.multiply(I1, Kx) - Vvx_hat + np.multiply(I2, Kx) - nu*np.multiply(K2, Vx_hat)
    Hy = F2_hat - np.multiply(I1, Ky) - Vvy_hat + np.multiply(I2, Ky) - nu*np.multiply(K2, Vy_hat)

    return Hx, Hy

def init_tracers(num_tracers):
    z = []
    for i in range(num_tracers):
        z.append([np.random.uniform(0, 1, 2)])
    return z

def get_velocity(loc):
    idxx1 = int(loc[0]//dx)
    idxy1 = int(loc[1]//dx)

    if idxx1 == N:
        idxx2 = 0
    else:
        idxx2 = idxx1+1

    if idxy1 == N:
        idxy2 = 0
    else:
        idxy2 = idxy1 + 1

    x1 = x[idxx1]
    x2 = x1+dx
    y1 = y[idxy1]
    y2 = y1+dx

    #bilinear interpolation
    interp2d = np.array([[x2*y2, -x2*y1, -x1*y2, x1*y1], [-y2, y1, y2, -y1], [-x2, x2, x1, -x1], [1, -1, -1, 1]])
    a = 1.0/dx/dx*np.dot(interp2d, np.array([vx[idxx1, idxy1], vx[idxx1, idxy2], vx[idxx2, idxy1], vx[idxx2, idxy2]]))
    u1 = np.dot(a, np.array([1, loc[0], loc[1], loc[0]*loc[1]]))
    a = 1.0/dx/dx*np.dot(interp2d, np.array([vy[idxx1, idxy1], vy[idxx1, idxy2], vy[idxx2, idxy1], vy[idxx2, idxy2]]))
    u2 = np.dot(a, np.array([1, loc[0], loc[1], loc[0]*loc[1]]))

    return np.array([u1, u2])

def update_tracer(z, dt):
    for tracer in z:
        velocity = get_velocity(tracer[-1])
        tracer_update = tracer[-1]+velocity*dt
        if tracer_update[0] >= 1.0:
            tracer_update[0] = tracer_update[0] - 1.0
        if tracer_update[0] < 0.0:
            tracer_update[0] = tracer_update[0] + 1.0
        if tracer_update[1] >= 1.0:
            tracer_update[1] = tracer_update[1] - 1.0
        if tracer_update[1] < 0.0:
            tracer_update[1] = tracer_update[1] + 1.0
        tracer.append(tracer_update)
    return z

def RK4(t0, tf, ds):
    global Vx_hat, Vy_hat, vx, vy
    t = t0

    fig, ax = plt.subplots(1, 2)
    cax1 = make_axes_locatable(ax[0]).append_axes("right", size="5%", pad="2%")
    cax2 = make_axes_locatable(ax[1]).append_axes("right", size="5%", pad="2%")

    z = init_tracers(5)

    while t < tf - 0.5*ds:
        print(t)

        Hx, Hy = rhs(m, t)
        Qx = Hx*ds
        Qy = Hy*ds
        Vx_hat = Vx_hat + 0.1028639988105*Qx
        Vy_hat = Vy_hat + 0.1028639988105*Qy
            
        Hx, Hy = rhs(m, t+0.1028639988105*ds)
        Qx = -0.4801594388478*Qx + Hx*ds
        Qy = -0.4801594388478*Qy + Hy*ds
        Vx_hat = Vx_hat + 0.7408540575767*Qx
        Vy_hat = Vy_hat + 0.7408540575767*Qy
            
        Hx, Hy = rhs(m, t+0.487989987833*ds)
        Qx = -1.4042471952*Qx+Hx*ds
        Qy = -1.4042471952*Qy+Hy*ds
        Vx_hat = Vx_hat + 0.7426530946684*Qx
        Vy_hat = Vy_hat + 0.7426530946684*Qy

        Hx, Hy = rhs(m, t+0.6885177231562*ds)
        Qx = -2.016477077503*Qx+Hx*ds
        Qy = -2.016477077503*Qy+Hy*ds
        Vx_hat = Vx_hat + 0.4694937902358*Qx
        Vy_hat = Vy_hat + 0.4694937902358*Qy
            
        Hx, Hy = rhs(m, t+0.9023816453077*ds)
        Qx = -1.056444269767*Qx+Hx*ds
        Qy = -1.056444269767*Qy+Hy*ds
        Vx_hat = Vx_hat + 0.1881733382888*Qx
        Vy_hat = Vy_hat + 0.1881733382888*Qy        

        vx = np.real(np.fft.ifft2(Vx_hat))
        vy = np.real(np.fft.ifft2(Vy_hat))

        update_tracer(z, ds)
        t = t+ds

        if int(t*10000) % 100 == 0:
            vx_plot=ax[0].contourf(X, Y, vx)
            vy_plot=ax[1].contourf(X, Y, vy)
            fig.colorbar(vx_plot, cax=cax1)
            fig.colorbar(vy_plot, cax=cax2)
            for tracers in z:
                z_array = np.array(tracers)
                ax[0].plot(z_array[:, 0], z_array[:, 1], "ro", linestyle = 'None')
                ax[1].plot(z_array[:, 0], z_array[:, 1], "ro", linestyle = 'None')
            plt.pause(0.0001)


def int2d_newton_cotes(N, power):
    NC=7  #need to be odd number
    Nn=999 #need to be odd number
    Ntotal=Nn*NC+1
    dL=1.0/(Ntotal-1)
    xint=np.linspace(0, 1, Ntotal,endpoint=True)
    Xint, Yint = np.meshgrid(xint, xint)

    # vxxHat = 1j*np.multiply(Kx, Vx_hat)
    vxyHat = 1j*np.multiply(Ky, Vx_hat)
    vyxHat = 1j*np.multiply(Kx, Vy_hat)
    # vyyHat = 1j*np.multiply(Ky, Vy_hat)

    # vxxHat = np.fft.fftshift(vxxHat)
    vxyHat = np.fft.fftshift(vxyHat)
    vyxHat = np.fft.fftshift(vyxHat)
    # vyyHat = np.fft.fftshift(vyyHat)

    pad_size = int((Ntotal-N-2)/2)
    # vxxHat = np.pad( vxxHat, (((Ntotal-N-2)/2, (Ntotal-N-2)/2), ((Ntotal-N-2)/2, (Ntotal-N-2)/2)), 'constant', constant_values=0) 
    vxyHat = np.pad( vxyHat, ((pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=0)
    vyxHat = np.pad( vyxHat, ((pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=0)
    # vyyHat = np.pad( vyyHat, (((Ntotal-N-2)/2, (Ntotal-N-2)/2), ((Ntotal-N-2)/2, (Ntotal-N-2)/2)), 'constant', constant_values=0)    

    # vxxHat = np.fft.ifftshift(vxxHat*((Ntotal)/(N+1))**2) 
    vxyHat = np.fft.ifftshift(vxyHat*((Ntotal-1)/(N+1))**2)
    vyxHat = np.fft.ifftshift(vyxHat*((Ntotal-1)/(N+1))**2)
    # vyyHat = np.fft.ifftshift(vyyHat*((Ntotal)/(N+1))**2)

    # vxxInt = np.real(np.fft.ifft2(vxxHat))
    vxyInt = np.real(np.fft.ifft2(vxyHat))
    vyxInt = np.real(np.fft.ifft2(vyxHat))
    # vyyInt = np.real(np.fft.ifft2(vyyHat))

    # vxxInt = np.pad(np.real(vxxInt), ((0,1), (0, 1)), 'reflect')
    vxyInt = np.pad(np.real(vxyInt), ((0,1), (0,1)), 'reflect')
    vyxInt = np.pad(np.real(vyxInt), ((0,1), (0,1)), 'reflect')
    # vyyInt = np.pad(np.real(vyyInt), ((0,1), (0, 1)), 'reflect')

    an, b = newton_cotes(NC, 1)
    atmp = np.zeros(Ntotal)
    for i in range(math.floor(Ntotal/NC)):
        atmp[i*NC:(i+1)*NC+1] += an*dL
    weights = np.dot(np.transpose([atmp]), [atmp])
    coordinatePow = np.multiply(np.power(Xint, power), np.power(Yint, power))
    vxIn = np.multiply(coordinatePow, vxyInt)
    vyIn = np.multiply(coordinatePow, vyxInt)
    vIn = vxIn - vyIn

    return 100*np.sum(np.multiply(weights, vIn))

def solve_NS(sample):
    global m, nu, N
    m = sample
    nu = 0.1
    N = 128
    init_mesh(N)

    t0 = 0.0
    tf = 5.0
    ds = 0.0001
    RK4(t0, tf, ds)

    results = int2d_newton_cotes(N, 0.5)
    print(results)


if __name__ == "__main__":
    solve_NS(1.0)

    # fig, ax = plt.subplots(1, 2)
    # cax1 = make_axes_locatable(ax[0]).append_axes("right", size="5%", pad="2%")
    # cax2 = make_axes_locatable(ax[1]).append_axes("right", size="5%", pad="2%")

    # vx_plot=ax[0].contourf(X, Y, vx)
    # vy_plot=ax[1].contourf(X, Y, vy)
    # fig.colorbar(vx_plot, cax=cax1)
    # fig.colorbar(vy_plot, cax=cax2)
    # for tracers in z:
    #     z_array = np.array(tracers)
    #     ax[0].plot(z_array[:, 0], z_array[:, 1], "ro", linestyle = 'None')
    #     ax[1].plot(z_array[:, 0], z_array[:, 1], "ro", linestyle = 'None')
    # plt.show()
