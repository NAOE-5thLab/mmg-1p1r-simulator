import numpy as np
import sys

def get_rudderforce(u_velo, vm_velo, r_angvelo, delta_rudder, n_prop,
                    one_minus_wprop, t_prop, J_prop,
                    rho_fresh, lpp, dia_prop, area_rudder, lambda_rudder, x_location_rudder , 
                    t_rudder, ah_rudder, xh_rudder_nd, lr_rudder_nd, kx_rudder, epsilon_rudder,
                    gammaP, gammaN,
                    kx_rudder_reverse, cpr_rudder, coeff_urudder_zero,
                    XP, switch_rudder,switch_fn_rudder,switch_ur_rudder):
    ##########################
    ### Force of Ruder ###
    ##########################
    uprop = u_velo * one_minus_wprop
    xH = xh_rudder_nd * lpp
    
    aR, resultant_U_rudder, u_rudder, v_rudder = get_effective_inflow_ang(u_velo, vm_velo, r_angvelo, delta_rudder, n_prop, 
                    one_minus_wprop, uprop, t_prop, J_prop,
                    rho_fresh, lpp, dia_prop, area_rudder, lambda_rudder, x_location_rudder , 
                    lr_rudder_nd, kx_rudder, epsilon_rudder, 
                    gammaP, gammaN,
                    kx_rudder_reverse, cpr_rudder, coeff_urudder_zero,
                    XP,switch_ur_rudder)
    
    if switch_fn_rudder ==1:
        coeff_n = get_cn_fujii(lambda_rudder, aR)
        coeff_t = np.zeros([coeff_n.size,1])
    elif switch_fn_rudder==2:
        coeff_n,coeff_t = get_cnct_lindenburg(lambda_rudder, aR)
    else:
        print('switch_fn error')
        sys.exit()
        
    Force_normal = 0.5 * rho_fresh * area_rudder * resultant_U_rudder**2 * coeff_n 
    Force_tangental = 0.5 * rho_fresh * area_rudder * resultant_U_rudder**2 * coeff_t
    
    ### compute each componet of rudder force
    if switch_rudder == 1:
        XR = -(1 - t_rudder)                       * (Force_normal * np.sin(delta_rudder) + Force_tangental* np.cos(delta_rudder)) 
        YR = -(1 + ah_rudder)                      * (Force_normal * np.cos(delta_rudder) - Force_tangental* np.sin(delta_rudder))
        NR = -(x_location_rudder + ah_rudder * xH) * (Force_normal * np.cos(delta_rudder) - Force_tangental* np.sin(delta_rudder))
    
    ###  original rudder model (XR=YR=NR=0 at n<0) 
    elif switch_rudder == 0:
        XR = np.where(n_prop > 0,  -(1 - t_rudder) * Force_normal * np.sin(delta_rudder), np.array(0.0))
        YR = np.where(n_prop > 0,  -(1 + ah_rudder)* Force_normal * np.cos(delta_rudder), np.array(0.0))
        NR = np.where(n_prop > 0,  -(x_location_rudder + ah_rudder * xH) * Force_normal * np.cos(delta_rudder), np.array(0.0))
        
    rudderforce = np.concatenate((XR, YR, NR, aR, resultant_U_rudder, u_rudder, v_rudder), axis=1)
    return rudderforce

def get_effective_inflow_ang(u_velo, vm_velo, r_angvelo, delta_rudder, n_prop, 
                    one_minus_wprop, uprop, t_prop, J_prop,
                    rho_fresh, lpp, dia_prop, area_rudder, lambda_rudder, x_location_rudder , 
                    lr_rudder_nd, kx_rudder, epsilon_rudder, 
                    gammaP, gammaN,
                    kx_rudder_reverse, cpr_rudder, coeff_urudder_zero,
                    XP,switch_ur_rudder):
    
    # Calculate effective velocity to rudder uR
    hight_rudder = np.sqrt(area_rudder * lambda_rudder)
    eta_rudder = dia_prop / hight_rudder
    lR = lr_rudder_nd * lpp
    kappa_rudder = kx_rudder / epsilon_rudder      
    one_minus_wrudder = epsilon_rudder * one_minus_wprop
    ### Kitagawa's model for uR in n<0
    KT = np.where( n_prop > np.finfo(float).eps,  # n>0
                    XP / (rho_fresh * dia_prop**4 * n_prop**2 * (1-t_prop)), 
                    np.where(np.abs(n_prop) > np.finfo(float).eps, #n<0
                                XP / (rho_fresh * dia_prop**4 * n_prop**2),
                                np.array(0.0) # n = 0
                                )
                    )
    urpr1 = u_velo * one_minus_wrudder + n_prop * dia_prop * kx_rudder_reverse * np.sqrt(8 * np.abs(KT)/ np.pi)
    urpr2 = u_velo * one_minus_wrudder
    ursq  = eta_rudder * np.sign(urpr1) * urpr1**2 + (1- eta_rudder) * urpr2**2 + cpr_rudder * u_velo**2
    if switch_ur_rudder == 1:
        uR = np.where((n_prop >= 0)&(KT > 0), 
                        epsilon_rudder * np.sqrt(eta_rudder 
                        * (uprop+ kappa_rudder * (np.sqrt(uprop**2 + 8 * KT * n_prop**2 * dia_prop**2/ (np.pi )) - uprop))**2 + (1- eta_rudder) * uprop**2), # <= normarl mmg model for low speed (Yoshimura's)
                        # uprop * epsilon_rudder * np.sqrt(eta_rudder 
                        # * (1.0 + kappa_rudder * (np.sqrt(1 + 8 * KT / (np.pi * J_prop**2)) - 1.0))**2 + (1- eta_rudder)), # <= normarl mmg model
                        np.where( u_velo >= 0,
                                np.sign(ursq) * np.sqrt(np.abs(ursq)),  # <= kitagawa's model for n<0 (3rd q)
                                u_velo # <= uR=u at 4th quadrant
                            )
                    )### note:J_prop=Js at backward !!!!)
    elif switch_ur_rudder ==2:
        uR_star =  uprop * epsilon_rudder * (eta_rudder *  kappa_rudder * 
                                            (np.sign(u_velo)*np.sqrt(1 + 8.0 * KT / (np.pi * J_prop**2)) - 1.0) + 1.0) 
        uR_twostar = 0.7 * np.pi * n_prop * dia_prop * coeff_urudder_zero
        uR = np.where((n_prop >= 0)&(KT > 0), 
                np.where( u_velo>=0,
                         epsilon_rudder * np.sqrt(eta_rudder * (uprop+ kappa_rudder * 
                        (np.sqrt(uprop**2 + 8 * KT * n_prop**2 * dia_prop**2/ (np.pi )) - uprop))**2 + (1- eta_rudder) * uprop**2), # <= 1st q, normarl mmg model for low speed (Yoshimura's)
                        np.where((uR_twostar-uR_star) * np.sign(u_velo) < 0.0,  # <= 2nd q, Yasukawa's model for backward
                                uR_star,
                                uR_twostar)),
                np.where( u_velo >= 0,
                        np.sign(ursq) * np.sqrt(np.abs(ursq)),  # <= kitagawa's model for n<0 (3rd q)
                        u_velo # <= uR=u at 4th quadrant
                    )
            )### note:J_prop=Js at backward !!!!)
    else:
        print('switch_ur error')
        sys.exit()
                                                                
    vR = np.where(vm_velo + x_location_rudder * r_angvelo >= 0, 
                  -1.0*gammaP * (vm_velo + lR * r_angvelo), 
                  -1.0*gammaN * (vm_velo + lR * r_angvelo))
    resultant_U_rudder = np.sqrt(uR**2 + vR**2)
    aR = delta_rudder - np.arctan2(vR, uR)
    return aR, resultant_U_rudder, uR, vR

def get_cn_fujii(lambda_rudder, AngleOfAttack):
    return  6.13 * lambda_rudder / (2.25 + lambda_rudder) *np.sin(AngleOfAttack)

def get_cnct_lindenburg(lambda_rudder, AngleOfAttack, max_tickness_rudder=0.18):
    coeff_n =  1.98 * \
                            (1.0/(0.56+0.44 * np.abs(np.sin(AngleOfAttack)))-0.41*(1.0-np.exp(-17/lambda_rudder))) \
                            * np.sin(AngleOfAttack)
    r_nose_by_c = 1.1019 * max_tickness_rudder **2.0
    gamma_tangental = 0.28 * np.sqrt(r_nose_by_c)
    
    coeff_t= 0.5*0.0075*np.cos(AngleOfAttack)+coeff_n * (1.0/np.tan(np.pi/2.0+gamma_tangental))                        
    return coeff_n, coeff_t