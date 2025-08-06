import numpy as np

def vectorize_principal_particulars(pp_df):
    """ convert pp from dataframe to ndarray
    
    pp contains geometric constant of subject ship, loaded from csv file
    Args:
        pp_df (dataflame, float): set of principle parameters in DATAFRAME form.
    Returns:
        pp_vector (ndarray, float): set of principal parameters in NDARRAY form
    """
    pp_vector = np.empty(len(pp_df))
    ### Principal Particulars ###
    pp_vector[0]  = pp_df.at['lpp','value'] 
    pp_vector[1]  = pp_df.at['breadth','value'] 
    pp_vector[2]  = pp_df.at['draft','value'] 
    pp_vector[3]  = pp_df.at['mass_nd','value'] 
    pp_vector[4]  = pp_df.at['x_lcg','value'] 
    # Propeller
    pp_vector[5]       = pp_df.at['dia_prop','value'] 
    pp_vector[6]       = pp_df.at['pitchr','value']  
    # Rudder
    pp_vector[7]  = pp_df.at['area_rudder','value'] 
    pp_vector[8]  = pp_df.at['lambda_rudder','value'] 
    pp_vector[9]  = pp_df.at['x_location_rudder','value']
    # Side thrusters
    pp_vector[10]  = pp_df.at['D_bthrust','value'] 
    pp_vector[11]  = pp_df.at['D_sthrust','value'] 
    pp_vector[12]  = pp_df.at['x_bowthrust_loc','value']
    pp_vector[13]  = pp_df.at['x_stnthrust_loc','value']
    pp_vector[14]  = pp_df.at['Thruster_speed_max','value']
    # parameters for wind force computation
    pp_vector[15] = pp_df.at['area_projected_trans','value']  # AT
    pp_vector[16] = pp_df.at['area_projected_lateral','value']  # AL
    pp_vector[17] = pp_df.at['AOD','value']  # AOD 
    pp_vector[18] = pp_df.at['LCW','value']  # C or LCW, midship to center of AL
    pp_vector[19] = pp_df.at['LCBR','value']  # CBR, midship to center of AOD(superstructure or bridge)
    pp_vector[20] = pp_df.at['HBR','value'] #  Height from free surface to top of the superstructure (bridge) (m) 
    pp_vector[21] = pp_df.at['HC','value']  # Hc, hight of center of lateral projected area
    pp_vector[22] = pp_df.at['SBW','value']  # ship breadth fro wind force computation 
    pp_vector[23] = pp_df.at['Lz','value']  # Lz --- Acting position of sway force from center of gravity (it is necessary to calculate roll moment)

    return pp_vector
    
def vectorize_mmg_params(mmg_params_df):
    """ convert mmg_params from dataframe to ndarray
    
    mmg params contains hydroderivative of mmg model, loaded from csv file
    Args:
        pp_df (dataflame, float): set of principle parameters in DATAFRAME form.
    Returns:
        pp_vector (ndarray, float): set of principal parameters in NDARRAY form
    """
    mmg_params_vector = np.empty(len(mmg_params_df))    
    ### Parameter Init ###
    mmg_params_vector[0]      = mmg_params_df.at['massx_nd','value'] 
    mmg_params_vector[1]      = mmg_params_df.at['massy_nd','value'] 
    mmg_params_vector[2]      = mmg_params_df.at['IzzJzz_nd','value'] 
    # Hull
    mmg_params_vector[3]       = mmg_params_df.at['xuu_nd','value'] 
    mmg_params_vector[4]       = mmg_params_df.at['xvr_nd','value'] 
    mmg_params_vector[5]       = mmg_params_df.at['yv_nd','value'] 
    mmg_params_vector[6]       = mmg_params_df.at['yr_nd','value']
    mmg_params_vector[7]       = mmg_params_df.at['nv_nd','value'] 
    mmg_params_vector[8]       = mmg_params_df.at['nr_nd','value'] 
    mmg_params_vector[9]       = mmg_params_df.at['coeff_drag_sway','value']
    mmg_params_vector[10]       = mmg_params_df.at['cry_cross_flow','value'] 
    mmg_params_vector[11]       = mmg_params_df.at['crn_cross_flow','value'] 
    mmg_params_vector[12]       = mmg_params_df.at['coeff_drag_aft','value']
    # Propeller
    mmg_params_vector[13]      = mmg_params_df.at['t_prop','value']  
    mmg_params_vector[14]      = mmg_params_df.at['w_prop_zero','value']  
    mmg_params_vector[15]      = mmg_params_df.at['tau_prop','value'] 
    mmg_params_vector[16]      = mmg_params_df.at['coeff_cp_prop','value'] 
    mmg_params_vector[17]      = mmg_params_df.at['xp_prop_nd','value'] 
    mmg_params_vector[18:21]      = np.array([mmg_params_df.at['kt_coeff0','value'], mmg_params_df.at['kt_coeff1','value'], 
                                    mmg_params_df.at['kt_coeff2','value']]) 
    mmg_params_vector[21:29]           = np.array([mmg_params_df.at['ai_coeff_prop0','value'], mmg_params_df.at['ai_coeff_prop1','value'], 
                                    mmg_params_df.at['ai_coeff_prop2','value'], mmg_params_df.at['ai_coeff_prop3','value'],
                                    mmg_params_df.at['ai_coeff_prop4','value'], mmg_params_df.at['ai_coeff_prop5','value'],
                                    mmg_params_df.at['ai_coeff_prop6','value'], mmg_params_df.at['ai_coeff_prop7','value']]) 
    mmg_params_vector[29:37]           = np.array([mmg_params_df.at['bi_coeff_prop0','value'], mmg_params_df.at['bi_coeff_prop1','value'], 
                                    mmg_params_df.at['bi_coeff_prop2','value'], mmg_params_df.at['bi_coeff_prop3','value'],
                                    mmg_params_df.at['bi_coeff_prop4','value'], mmg_params_df.at['bi_coeff_prop5','value'],
                                    mmg_params_df.at['bi_coeff_prop6','value'], mmg_params_df.at['bi_coeff_prop7','value']])
    mmg_params_vector[37:41]            = np.array([mmg_params_df.at['ci_coeff_prop0','value'], mmg_params_df.at['ci_coeff_prop1','value'], 
                                    mmg_params_df.at['ci_coeff_prop2','value'], mmg_params_df.at['ci_coeff_prop3','value']] )

    # Rudder
    mmg_params_vector[41]  = mmg_params_df.at['t_rudder','value'] 
    mmg_params_vector[42]  = mmg_params_df.at['ah_rudder','value']
    mmg_params_vector[43]  = mmg_params_df.at['xh_rudder_nd','value']
    mmg_params_vector[44]  = mmg_params_df.at['kx_rudder','value']
    mmg_params_vector[45]  = mmg_params_df.at['epsilon_rudder','value'] 
    mmg_params_vector[46]  = mmg_params_df.at['lr_rudder_nd','value']
    mmg_params_vector[47]  = mmg_params_df.at['gammaN_rudder','value']
    mmg_params_vector[48]  = mmg_params_df.at['gammaP_rudder','value']
    mmg_params_vector[49]  = mmg_params_df.at['kx_rudder_reverse','value']
    mmg_params_vector[50]  = mmg_params_df.at['cpr_rudder','value'] 
    mmg_params_vector[51]  = mmg_params_df.at['KT_bow_forward','value'] 
    mmg_params_vector[52]  = mmg_params_df.at['KT_bow_reverse','value'] 
    mmg_params_vector[53]  = mmg_params_df.at['aY_bow','value'] 
    mmg_params_vector[54]  = mmg_params_df.at['aN_bow','value'] 
    mmg_params_vector[55]  = mmg_params_df.at['KT_stern_forward','value'] 
    mmg_params_vector[56]  = mmg_params_df.at['KT_stern_reverse','value']
    mmg_params_vector[57]  = mmg_params_df.at['aY_stern','value'] 
    mmg_params_vector[58]  = mmg_params_df.at['aN_stern','value']  

    mmg_params_vector[59] = mmg_params_df.at['XX0','value'] 
    mmg_params_vector[60] = mmg_params_df.at['XX1','value'] 
    mmg_params_vector[61] = mmg_params_df.at['XX3','value'] 
    mmg_params_vector[62] = mmg_params_df.at['XX5','value'] 
    mmg_params_vector[63] = mmg_params_df.at['YY1','value'] 
    mmg_params_vector[64] = mmg_params_df.at['YY3','value']
    mmg_params_vector[65] = mmg_params_df.at['YY5','value']
    mmg_params_vector[66] = mmg_params_df.at['NN1','value']  
    mmg_params_vector[67] = mmg_params_df.at['NN2','value']  
    mmg_params_vector[68] = mmg_params_df.at['NN3','value']
    
    return mmg_params_vector