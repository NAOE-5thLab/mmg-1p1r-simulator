import numpy as np
import math
# from numba import jit

# @jit('f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:])',nopython=True)
def true2Apparent(UT, U_ship, zeta_wind, beta_hat, psi):
    """return relative (apparent) direction and speed of wind or current
    
    return relative dir/speed from earth fix (true) wind/current from true wind/current.
    process several timestep's data at once.(use np.ndarray)
    All Args and Returns are :x.shape = [number_of_timestep] 
    Args:
        UT (float, ndarray): velocity norm of true wind (m/s) UT>0
        U_ship (float, ndarray): velocity norm of ship  (m/s) U_ship>0
        zeta_wind (float, ndarray): direction of true wind, earth fix coordinate (!! rad !!),
        [0, 2pi], clock-wise.
        beta_hat (float, ndarray): angle from heading to COG (=-beta) (!! rad !!) (-pi, pi)
        psi (float, ndarray): heading of ship (!! rad !!) (-pi, pi), clock-wise.
    Returns:
        UA (float, ndarray): Relative speed of Wind or Current to ship(m/s) 
        gamma_a  (float, ndarray): Relative direction of Wind or Current to ship, 
        =angle between ship heading and relative wind (!! rad !!) [0, 2pi], clock-wise
    """
    UA = np.empty((UT.shape[0],1),dtype=np.float64)
    gamma_a = np.empty((UT.shape[0],1),dtype=np.float64)
    relativew = np.empty((UT.shape[0],2),dtype=np.float64)
    beta_hat = np.where(beta_hat < 0, beta_hat + 2*np.pi, beta_hat)
    _chi  = (3*np.pi - zeta_wind + psi) % (2*np.pi)
    
    # compute chi_beta(angle between ship traveling direction and true wind)
    chi_beta = _chi + beta_hat
    chi_beta = np.where(chi_beta < 0,
                        chi_beta + 2.0*np.pi,
                        np.where(chi_beta > 2.0*np.pi,
                                chi_beta - 2.0*np.pi,
                                chi_beta))
    
    # compute apparent wind speed
    UA = np.where((UT==0.0) &(U_ship==0.0),
                0.0, 
                np.sqrt(UT**2 + U_ship**2 - 2.0*UT*U_ship*np.cos(chi_beta))
        )
    U_ship = np.where(
       np.abs(U_ship)<1e-8,
       1e-8,
       U_ship 
    )
    # use intermidiate value rv To avoid the value larger than 1.0 due to numerical error on acos. 
    _rv = np.where(
        (UT==0.0) &(U_ship==0.0),
        0.0,
        (U_ship**2 + UA**2 - UT**2) / (2.0 * U_ship * UA)
    )
    rv = np.where(np.abs(_rv) > 1.0, np.sign(_rv), _rv)
    
    # compute gamma_beta(angle between ship traveling direction and apparent wind)
    # use if sentence because acos range is (0 pi) and gamma_beta (0 2pi)
    gamma_beta = np.where(chi_beta <= np.pi,
                            np.arccos(rv),
                            2.0*np.pi - np.arccos(rv)
                            )
    gamma_a = gamma_beta + beta_hat
    gamma_a = np.where(np.abs(U_ship)< 1.0e-5,
                        (3*np.pi - _chi) % (2*np.pi),
                        gamma_a
                        )
    gamma_a = np.where(np.abs(UT) < 1.0e-5,
                        beta_hat,
                        gamma_a
                        )
    gamma_a = np.where(gamma_a > 2.0*np.pi,
                        gamma_a - 2.0*np.pi,
                        gamma_a
                        )
    gamma_a = np.where(gamma_a < 0.0,
                        gamma_a + 2.0*np.pi,
                        gamma_a
                        )
    relativew = np.concatenate((UA, gamma_a), axis=1)
    return relativew


def apparent2True(UA, U_ship, gamma_a, beta_hat, psi):
    """return true direction and speed of wind or current
    
    return earth fix (true) dir/speed of wind/current from apparent (relative) wind/current.
    process several timestep's data at once.(use np.ndarray).
    All Args and Returns are :x.shape = [number_of_timestep] 
    Args:
        UA (float, ndarray): Relative speed of Wind or Current to ship(m/s) 
        U_ship (float, ndarray): velocity norm of ship  (m/s) U_ship>0
        gamma_a  (float, ndarray): Relative direction of Wind or Current to ship, 
        =angle between ship heading and relative wind (!! rad !!) [0, 2pi], clock-wise
        beta_hat (float, ndarray): drift angle of ship (!! rad !!) [-pi, pi]
        psi (float, ndarray): heading of ship (!! rad !!) [-pi, pi]
    Returns:
        UT (float, ndarray): velocity norm of true wind (m/s) UT>0
        zeta_wind (float, ndarray): direction of true wind, earth fix coordinate (!! rad !!),
        [0, 2pi], clock-wise
    """
    beta_hat = np.where(beta_hat < 0, beta_hat + 2*np.pi, beta_hat)
    # compute gamma_beta(angle between ship traveling direction and apparent wind direction gamma_a)
    gamma_beta = gamma_a-beta_hat
    gamma_beta = np.where(gamma_beta < 0,
                        gamma_beta + 2.0*np.pi,
                        np.where(gamma_beta > 2.0*np.pi,
                                gamma_beta - 2.0*np.pi,
                                gamma_beta))
    # compute true wind speed
    UT = np.where(
        (UA==0.0)&(U_ship==0.0),
        0.0,
        np.sqrt(UA**2 + U_ship**2 - 2.0*UA*U_ship*np.cos(gamma_beta))
        )
    
    # use intermidiate value rv To avoid the value larger than 1.0 due to numerical error on acos. 
    _rv = (U_ship**2 + UT**2 - UA**2) / (2.0 * U_ship * UT)
    rv = np.where(np.abs(_rv) > 1.0, np.sign(_rv), _rv)
    
    # compute chi_beta(angle between ship traveling direction and true wind)
    # use if sentence because acos range is (0 pi) and chi_beta (0 2pi)
    chi_beta = np.where(gamma_beta <= np.pi,
                            np.arccos(rv),
                            2.0*np.pi - np.arccos(rv)
                            )
    
    zeta_wind = psi-chi_beta + beta_hat + np.pi
    zeta_wind = np.where(np.abs(UT) < 1.0e-5,
                        0.0,
                        zeta_wind
                        )
    zeta_wind = np.where(zeta_wind > 2.0*np.pi,
                        zeta_wind - 2.0*np.pi,
                        zeta_wind
                        )
    zeta_wind = np.where(zeta_wind < 0.0,
                        zeta_wind + 2.0*np.pi,
                        zeta_wind
                        )
    
    return np.concatenate([UT, zeta_wind], axis=1)

### debug
# val_no    = 9
# windv  = 2.0 * np.ones((val_no, 1))
# windd  = np.linspace(0, 2*np.pi, val_no)

# u = -1.0 * np.ones((val_no,1))
# vm = -1.0 * np.ones((val_no,1))
# shipv = np.sqrt(u**2 + vm**2)
# beta_hat = np.arctan2(vm, u)
# shipd = np.zeros((val_no,1))

# windd = np.reshape(windd, [np.size(windd), 1])
# shipd = np.reshape(shipd, [np.size(shipd), 1])

# test = true_to_apparent(windv, shipv, windd, beta_hat, shipd)
# check = apparent_to_true(test[:,0:1], shipv, test[:,1:2], beta_hat, shipd)
# print(test, check)