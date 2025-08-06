import numpy as np
import time
def rk4_mmg(dt, function, time, stateval, n_prop, delta_rudder, wind_velo_true, wind_dir_true):
    """ classical Runge-kutta method subroutine for MMG model
    
    return slope of diff. eq. by 4th order Runge-Kutta method for mmg model.

    Args:
        dt (float): integral period of ODE
        function : function which will be integrated. mainly, MMG hydroForce funcion.
        time(float, ndarray): physical time of simulation. use for wind computation.
        stateval(float, ndarray): state values of ODE. derivative of stateval should be equal with the return of funcion.
        n_prop(float,ndarray): propeller rev.
        delta_rudder(float,ndarray): rudder angle (!! rad !!)
        wind_velo_true(float,ndarray): true wind velocity(wind velocity to ground), (m/s)
        wind_dir_true(float,ndarray): true wind velocity(wind velocity to ground), (m/s)
    Returns:
        slope (float, ndarray): time derivative of state value computed by 4th order Runge-Kutta method. 
    """
    save_intermid=True
    k1 = function(time, stateval, n_prop, delta_rudder, wind_velo_true, wind_dir_true,save_intermid)
    save_intermid=False
    k2 = function(time+0.5*dt, stateval + 0.5 * k1 * dt, n_prop, 
                            delta_rudder, wind_velo_true, wind_dir_true,save_intermid)
    k3 = function(time+0.5*dt, stateval + 0.5 * k2 * dt, n_prop, 
                            delta_rudder, wind_velo_true, wind_dir_true,save_intermid)
    k4 = function(time+1.0*dt, stateval + 1.0 * k3 * dt, n_prop, 
                            delta_rudder, wind_velo_true, wind_dir_true,save_intermid)
    slope = (k1 + 2*k2 + 2*k3 + k4)/6.0 
    return slope

def rk4_mmg_1P1R_BTST(dt, function, stateval, delta_rudder, n_prop, n_bt, n_st, wind_dir_true, wind_velo_true):
    """ classical Runge-kutta method subroutine for MMG model of 1P1R ship with bow & stern thruster
    
    return slope of diff. eq. by 4th order Runge-Kutta method for mmg model.
    for mmg module without irregular wind generation

    Args:
        dt (float): integral period of ODE
        function : function which will be integrated. mainly, MMG hydroForce funcion.
        stateval(float, ndarray): state values of ODE. derivative of stateval should be equal with the return of funcion.
        delta_rudder(float,ndarray): rudder angle (!! rad !!)
        n_prop(float,ndarray): propeller rev.
        n_bt(float,ndarray): rev. of bow thruster (rps)
        n_st(float,ndarray): rev. of stern thruster (rps)
        wind_dir_true(float,ndarray): true wind velocity(wind velocity to ground), (m/s)
        wind_velo_true(float,ndarray): true wind velocity(wind velocity to ground), (m/s)
    Returns:
        slope (float, ndarray): time derivative of state value computed by 4th order Runge-Kutta method. 
    """
    
    k1 = function(stateval, delta_rudder, n_prop, n_bt, n_st, wind_dir_true, wind_velo_true)
    k2 = function(stateval + 0.5 * k1 * dt, 
                            delta_rudder, n_prop, n_bt, n_st, wind_dir_true, wind_velo_true)
    k3 = function(stateval + 0.5 * k2 * dt,
                            delta_rudder, n_prop, n_bt, n_st, wind_dir_true, wind_velo_true)
    k4 = function(stateval + 1.0 * k3 * dt, 
                            delta_rudder, n_prop, n_bt, n_st, wind_dir_true, wind_velo_true)
    slope = (k1 + 2*k2 + 2*k3 + k4)/6.0 
    return slope

def rk4_mmg_test(dt, time, stateval, n_prop, delta_rudder, wind_velo_true, wind_dir_true):
    """ speed test code for RK4.
    do not use for practical purpose.
    """
    k1 = stateval * 0.1
    k2 = k1 *10.0
    k3 = k2 *0.1
    k4 = k3 *10
    slope = (k1+k2+k3+k4)/4.0
    return slope

def rk4_loop(no_timestep, dt_sec, function, physical_time, stateval, n_prop, delta_rudder, windv, windd):
    """ speed test code for RK4 and for sentence.
    do not use for practical purpose.
    """
    # start = time.time()
    for i in range(no_timestep-1):
        slope = rk4_mmg(dt_sec, function, physical_time[i:i+1], stateval[i:i+1,:], n_prop[i:i+1], 
                        delta_rudder[i:i+1], windv[i:i+1], windd[i:i+1])
        stateval[i+1:i+2, :] = stateval[i:i+1, :] + slope[:] * dt_sec
        physical_time[i+1,] = (i+1)*dt_sec
    # elapsed_time = time.time()-start
    # print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return physical_time, stateval