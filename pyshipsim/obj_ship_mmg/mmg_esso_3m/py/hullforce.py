import numpy as np
from scipy import integrate
# from numba import jit

def integrand_y(x, vm_velo, r_angvelo, C_ry):
    return np.abs(vm_velo + C_ry * r_angvelo * x) * (vm_velo + C_ry * r_angvelo * x)

def integrand_n(x, vm_velo, r_angvelo, C_rn):
    return np.abs(vm_velo + C_rn * r_angvelo * x) * (vm_velo + C_rn * r_angvelo * x) * x

# @jit('f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8)',nopython=True)
def hullForceYoshimura(U_ship, u_velo, vm_velo, r_angvelo, beta, lpp, draft, rho_fresh, X_0F_nd, X_0A_nd, Xvr_nd,
                        Yv_nd, Yr_nd, Nv_nd, Nr_nd, C_rY, C_rN, CD):
    """ hull force module with numba for speedup
    """
    hullforce = np.empty((U_ship.shape[0],3),dtype=np.float64)
    i_max = 200
    Y_ad = np.zeros((vm_velo.shape[0],1))
    N_ad = np.zeros((vm_velo.shape[0],1))
    YHN = np.empty_like(Y_ad)
    NHN = np.empty_like(Y_ad)
    for j in range(vm_velo.size):
        for i in range(i_max-1):
            x0 = -0.5+(i)/i_max
            x1 = -0.5+(i+1)  /i_max
            comp0 = vm_velo[j,0] + C_rY * r_angvelo[j,0] * lpp * x0
            comp1 = vm_velo[j,0] + C_rY * r_angvelo[j,0] * lpp * x1
            comp2 = vm_velo[j,0] + C_rN * r_angvelo[j,0] * lpp * x0
            comp3 = vm_velo[j,0] + C_rN * r_angvelo[j,0] * lpp * x1
            Y_ad[j,0] = Y_ad[j,0] + 0.5 * ( np.abs(comp0) * comp0      + np.abs(comp1) * comp1      ) / i_max
            N_ad[j,0] = N_ad[j,0] + 0.5 * ( np.abs(comp2) * comp2 * x0 + np.abs(comp3) * comp3 * x1 ) / i_max
        YHN[j,0] = -CD * Y_ad[j,0]
        NHN[j,0] = -CD * N_ad[j,0]
    # Culculate Hull forces
    XH = 0.5 * rho_fresh * lpp    * draft * ( ( X_0F_nd + (X_0A_nd - X_0F_nd)*(np.abs(beta)/np.pi) )* u_velo * U_ship + Xvr_nd * vm_velo * r_angvelo * lpp )
    YH = 0.5 * rho_fresh * lpp    * draft * ( Yv_nd * vm_velo * np.abs(u_velo) + Yr_nd * r_angvelo * lpp * u_velo            + YHN)
    NH = 0.5 * rho_fresh * lpp**2 * draft * ( Nv_nd * vm_velo * u_velo            + Nr_nd * r_angvelo * lpp * np.abs(u_velo) + NHN)
    hullforce = np.concatenate((XH, YH, NH), axis=1)
    return hullforce

def hullForceYoshimuraSimpson(U_ship, u_velo, vm_velo, r_angvelo, beta, lpp, draft, rho_fresh, X_0F_nd, X_0A_nd, Xvr_nd,
                        Yv_nd, Yr_nd, Nv_nd, Nr_nd, C_rY, C_rN, CD):
    """ hull force module with out numba
    
    using Scipy's integral function for speedup (sims) 
    """
    x= np.linspace(-lpp/2, lpp/2, num=100, endpoint=True)
    y = np.empty((vm_velo.shape[0], x.shape[0]))
    n = np.empty((r_angvelo.shape[0], x.shape[0]))
    y_clossflow =np.empty((U_ship.shape[0],1))
    n_clossflow = np.empty((U_ship.shape[0],1))
    for i in range(vm_velo.size):
        y[i,:] = integrand_y(x, vm_velo[i,0], r_angvelo[i,0], C_rY)
        # y[i,:] = np.abs(vm_velo[i,0] + C_rY * r_angvelo[i,0] * x[:]) * (vm_velo[i,0] + C_rY * r_angvelo[i,0] * x[:])
        n[i,:] = integrand_n(x, vm_velo[i,0], r_angvelo[i,0], C_rN)
        # n[i,:] = np.abs(vm_velo[i,0] + C_rN * r_angvelo[i,0] * x[:]) * (vm_velo[i,0] + C_rN * r_angvelo[i,0] * x[:]) * x[:]
        y_clossflow[i] = integrate.simps(y[i,:],x)/lpp * CD
        n_clossflow[i] = integrate.simps(n[i,:],x)/(lpp**2) * CD
    XH = 0.5 * rho_fresh * lpp    * draft * ( ( X_0F_nd + (X_0A_nd - X_0F_nd)
                                                      *(np.abs(beta)/np.pi) )* u_velo * U_ship + Xvr_nd * vm_velo * r_angvelo * lpp )
    YH = 0.5 * rho_fresh * lpp    * draft * ( Yv_nd * vm_velo * np.abs(u_velo) + Yr_nd * r_angvelo * lpp * u_velo       - y_clossflow)
    NH = 0.5 * rho_fresh * lpp**2 * draft * ( Nv_nd * vm_velo * u_velo    + Nr_nd * r_angvelo * lpp * np.abs(u_velo) - n_clossflow)
    hullforce = np.concatenate([XH, YH, NH], axis=1)
    return hullforce