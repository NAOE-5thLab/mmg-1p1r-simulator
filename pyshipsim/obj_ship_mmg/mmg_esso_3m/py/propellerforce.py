import numpy as np

def get_propellerforce(u_velo, vm_velo, r_angvelo, n_prop, 
                    rho_fresh, lpp, draft, dia_prop, pitchr,  
                    t_prop, wP0, tau, CP_nd, xP_nd, kt_coeff, 
                    Ai, Bi, Ci, Jmin, alpha_p, 
                    switch_prop_x, switch_prop_y, switch_prop_n):
    ##########################
    ### Force of Propeller ###
    ##########################
    
    U_ship    = np.sqrt(u_velo**2 + vm_velo**2)
    v_nd = np.where(np.abs(U_ship) < 1.0e-5, 1.0e-7 * np.ones(vm_velo.shape), vm_velo / U_ship        )
    r_nd = np.where(np.abs(U_ship) < 1.0e-5, 1.0e-7 * np.ones(r_angvelo.shape),  r_angvelo * lpp / U_ship)
    U_ship         = np.where(np.abs(U_ship) < 1.0e-5, 1.0e-7 * np.ones(U_ship.shape),  U_ship             )
    
    Jsyn  = -0.35
    Jsyn0 = -0.06
    #Jst   = -0.6

    wP = wP0 - tau * np.abs(v_nd + xP_nd*r_nd) - CP_nd * (v_nd + xP_nd*r_nd)**2
    one_minus_wprop = np.where(u_velo > 0,
                                np.ones(wP.shape) - wP,
                                np.ones(wP.shape)
                                )
    one_minus_wprop = np.where(one_minus_wprop > 1.0e0 ,np.array(1.0) , one_minus_wprop)
    one_minus_wprop = np.where(one_minus_wprop <= 0.0e0 ,np.array(np.finfo(float).eps) , one_minus_wprop)
    

    Js = np.where(np.abs(n_prop) < np.finfo(float).eps,  # avoid 0 divid for J_prop at small
                        np.sign(u_velo) * 1.0e4, 
                        np.where( np.abs(u_velo) < np.finfo(float).eps, # avoid j==0 
                                    np.array(np.finfo(float).eps),
                                    u_velo / (dia_prop * n_prop)
                                    )
                        ) 
    J_prop  = np.where( u_velo >=0, Js * one_minus_wprop, Js) 
    
    if switch_prop_x == 1:
    # Hachii's model for effective thrust 
        KT = kt_coeff[0] + kt_coeff[1] * J_prop + kt_coeff[2] * J_prop**2
        XP = np.where(n_prop >= 0, 
                        rho_fresh * dia_prop**4 * n_prop**2 * (1-t_prop) * KT, 
                        np.where(Js >= Ci[3], 
                                rho_fresh * dia_prop**4 * n_prop**2 * (Ci[1] + Ci[2] * Js), 
                                rho_fresh * dia_prop**4 * n_prop**2 * Ci[0]
                                )
                    )
    elif switch_prop_x == 2:
    # Yasukawa's model for effective thrust
        beta_t   = 0.4811 * pitchr**2 - 1.1116 * pitchr -0.1404

        KT  = np.where( n_prop > 0,
                    np.where( J_prop > Jmin,
                            kt_coeff[0] + kt_coeff[1] * J_prop + kt_coeff[2] * J_prop**2, 
                            alpha_p * ( kt_coeff[1]* (J_prop-Jmin) - kt_coeff[2] 
                            * (J_prop - Jmin)** 2) +kt_coeff[0] + kt_coeff[1] * Jmin + kt_coeff[2] * Jmin**2
                            ),
                            np.where( J_prop > Jmin,
                                    beta_t * ( kt_coeff[0] + kt_coeff[1] * J_prop + kt_coeff[2] * J_prop**2), 
                                    beta_t * ( alpha_p * ( kt_coeff[1]* (J_prop-Jmin) - kt_coeff[2] 
                                    * (J_prop - Jmin)** 2) +kt_coeff[0] + kt_coeff[1] * Jmin + kt_coeff[2] * Jmin**2)
                                    )
                )
    
        XP = np.where( n_prop>=0,
                  rho_fresh * dia_prop**4 * n_prop**2 * (1-t_prop) * KT,
                  rho_fresh * dia_prop**4 * n_prop**2 * KT
                )
    ### Yp and Np               
    ### Ueno's experiment data of training vessel(Seiunmaru) for 2nd quadrant (u <0, n>0)
    ### Hachii's model for 3rd(u>0, n<0) & 4th quadrant (u<0, n<0)
    pitch  = dia_prop * pitchr
    if switch_prop_y == 1 :
        YP = np.where(n_prop >= 0,              
                        np.where(u_velo >=0,
                                np.array(0.0),
                                0.5 * rho_fresh * lpp * draft * (n_prop * pitch)**2 * (Ai[5] * Js**2 + Ai[6] * Js + Ai[7])
                            ),
                        np.where(Js < Jsyn,
                                0.5 * rho_fresh * lpp * draft * (n_prop * dia_prop)**2 * (Ai[2] + Ai[3] * Js), 
                                np.where(Jsyn0 < Js,
                                    0.5 * rho_fresh * lpp    * draft * (n_prop * dia_prop)**2 * Ai[4],
                                    0.5 * rho_fresh * lpp    * draft * (n_prop * dia_prop)**2 * (Ai[0] + Ai[1] * Js)
                                    )
                            )
                )
    elif switch_prop_y == 0 :
        YP = np.where(n_prop >= 0,
                        np.array(0.0),
                        np.where(Js < Jsyn,
                                0.5 * rho_fresh * lpp * draft * (n_prop * dia_prop)**2 * (Ai[2] + Ai[3] * Js), 
                                np.where(Jsyn0 < Js,
                                    0.5 * rho_fresh * lpp    * draft * (n_prop * dia_prop)**2 * Ai[4],
                                    0.5 * rho_fresh * lpp    * draft * (n_prop * dia_prop)**2 * (Ai[0] + Ai[1] * Js)
                                    )
                            )
                )           
    if switch_prop_n == 1:
        NP = np.where(n_prop >= 0,
                    np.where(u_velo >=0,
                                np.array(0.0),
                                0.5 * rho_fresh * lpp**2 * draft * (n_prop * pitch)**2 * (Bi[5] * Js**2 + Bi[6] * Js + Bi[7])
                            ),
                    np.where(Js < Jsyn,
                                0.5 * rho_fresh * lpp**2 * draft * (n_prop * dia_prop)**2 * (Bi[2] + Bi[3] * Js),
                                np.where(Jsyn0 < Js,
                                          0.5 * rho_fresh * lpp**2 * draft * (n_prop * dia_prop)**2 * Bi[4],
                                          0.5 * rho_fresh * lpp**2 * draft * (n_prop * dia_prop)**2 * (Bi[0] + Bi[1] * Js)
                                        )
                            )
                    )
    elif switch_prop_n == 0:
        NP = np.where(n_prop >= 0,
                    np.array(0.0),
                    np.where(Js < Jsyn,
                                0.5 * rho_fresh * lpp**2 * draft * (n_prop * dia_prop)**2 * (Bi[2] + Bi[3] * Js),
                                np.where(Jsyn0 < Js,
                                          0.5 * rho_fresh * lpp**2 * draft * (n_prop * dia_prop)**2 * Bi[4],
                                          0.5 * rho_fresh * lpp**2 * draft * (n_prop * dia_prop)**2 * (Bi[0] + Bi[1] * Js)
                                        )
                            )
                    )
        
    propforce = np.concatenate((XP, YP, NP, one_minus_wprop, J_prop), axis=1)
    return propforce