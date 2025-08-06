
import math
from sys import setdlopenflags
import numpy as np
from subroutine import relative 
from subroutine import wind
from subroutine import hullforce
from subroutine import propellerforce
from subroutine import rudderforce
import pandas as pd

class MmgModel():

    def __init__(self, init_const, init_parameter, model_switch):
        """ instantiate MmgModel
        
        input principal particulas, model setting of computation, initial value of parameters.
        Args:
            init_const (dataflame, float): set of principle parameters etc.
            init_parameters (dataflame, float): set of initial parameteres (hydro derivatives, etc)
            model_switch (dataflame, int): set of model setting of computation
        Returns:
            none
        """
        # white model 
        ### Principal Particulars ###
        self.lpp               = init_const.at['lpp','value'] 
        self.B               = init_const.at['breadth','value'] 
        self.draft               = init_const.at['draft','value'] 
        self.Mass_nd    = init_const.at['mass_nd','value'] 
        self.xG_nd      = init_const.at['x_lcg','value'] 
        self.rho_fresh       = init_const.at['rho_fresh','value'] /9.80665
        # Propeller
        self.dia_prop        = init_const.at['dia_prop','value'] 
        self.pitchr          = init_const.at['pitchr','value']  
        # Rudder
        self.area_rudder     = init_const.at['area_rudder','value'] 
        self.lambda_rudder   = init_const.at['lambda_rudder','value'] 
        self.x_location_rudder = init_const.at['x_location_rudder','value']
        self.maxtickness_rudder = init_const.at['maxtickness_rudder','value']
        
        # parameters for wind force computation
        self.rho_air                               = init_const.at['rho_air','value'] /9.80665# density of air
        self.area_projected_trans                  = init_const.at['area_projected_trans','value']  # AT
        self.area_projected_lateral                = init_const.at['area_projected_lateral','value']  # AL
        self.area_projected_lateral_superstructure = init_const.at['AOD','value']  # AOD 
        self.lcw                                   = init_const.at['LCW','value']  # C or LCW, midship to center of AL
        self.lcbr                                  = init_const.at['LCBR','value']  # CBR, midship to center of AOD(superstructure or bridge)
        self.hbr                                   = init_const.at['HBR','value'] #  Height from free surface to top of the superstructure (bridge) (m) 
        self.hc_AL                                 = init_const.at['HC','value']  # Hc, hight of center of lateral projected area
        self.breadth_wind                          = init_const.at['SBW','value']  # ship breadth fro wind force computation 
        self.swayforce_to_cg                       = init_const.at['Lz','value']  # Lz --- Acting position of sway force from center of gravity (it is necessary to calculate roll moment)
        
        ### Parameter Init ###
        self.MassX_nd      = init_parameter.at['massx_nd','params_init'] 
        self.MassY_nd      = init_parameter.at['massy_nd','params_init'] 
        self.IJzz_nd       = init_parameter.at['IzzJzz_nd','params_init'] 
        # Hull
        self.Xuu_nd        = init_parameter.at['xuu_nd','params_init'] 
        self.Xvr_nd        = init_parameter.at['xvr_nd','params_init'] 
        self.Yv_nd         = init_parameter.at['yv_nd','params_init'] 
        self.Yr_nd         = init_parameter.at['yr_nd','params_init']
        self.Nv_nd         = init_parameter.at['nv_nd','params_init'] 
        self.Nr_nd         = init_parameter.at['nr_nd','params_init'] 
        self.CD            = init_parameter.at['coeff_drag_sway','params_init']#0.500
        self.C_rY          = init_parameter.at['cry_cross_flow','params_init'] #1.00
        self.C_rN          = init_parameter.at['crn_cross_flow','params_init'] #0.50
        self.X_0F_nd       = self.Xuu_nd
        self.X_0A_nd       = init_parameter.at['coeff_drag_aft','params_init']
        # Propeller
        self.t_prop        = init_parameter.at['t_prop','params_init']  
        self.wP0           = init_parameter.at['w_prop_zero','params_init']  
        self.tau           = init_parameter.at['tau_prop','params_init'] 
        self.CP_nd         = init_parameter.at['coeff_cp_prop','params_init'] 
        self.xP_nd         = init_parameter.at['xp_prop_nd','params_init'] 
        self.kt_coeff      = np.array([init_parameter.at['kt_coeff0','params_init'], init_parameter.at['kt_coeff1','params_init'], 
                                        init_parameter.at['kt_coeff2','params_init']]) 
        self.Ai            = np.array([init_parameter.at['ai_coeff_prop0','params_init'], init_parameter.at['ai_coeff_prop1','params_init'], 
                                        init_parameter.at['ai_coeff_prop2','params_init'], init_parameter.at['ai_coeff_prop3','params_init'],
                                        init_parameter.at['ai_coeff_prop4','params_init'], init_parameter.at['ai_coeff_prop5','params_init'],
                                        init_parameter.at['ai_coeff_prop6','params_init'], init_parameter.at['ai_coeff_prop7','params_init']]) 
        self.Bi            = np.array([init_parameter.at['bi_coeff_prop0','params_init'], init_parameter.at['bi_coeff_prop1','params_init'], 
                                        init_parameter.at['bi_coeff_prop2','params_init'], init_parameter.at['bi_coeff_prop3','params_init'],
                                        init_parameter.at['bi_coeff_prop4','params_init'], init_parameter.at['bi_coeff_prop5','params_init'],
                                        init_parameter.at['bi_coeff_prop6','params_init'], init_parameter.at['bi_coeff_prop7','params_init']])
        self.Ci            = np.array([init_parameter.at['ci_coeff_prop0','params_init'], init_parameter.at['ci_coeff_prop1','params_init'], 
                                        init_parameter.at['ci_coeff_prop2','params_init'], init_parameter.at['ci_coeff_prop3','params_init']] )
        self.Jmin          = init_parameter.at['Jmin','params_init']  # -0.5 coeff. from exp
        self.alpha_p       = init_parameter.at['alpha_prop','params_init']  # 1.5  coeff. from exp   
        
        # Rudder
        self.t_rudder           = init_parameter.at['t_rudder','params_init'] 
        self.ah_rudder          = init_parameter.at['ah_rudder','params_init']
        self.xh_rudder_nd       = init_parameter.at['xh_rudder_nd','params_init'] 
        self.lr_rudder_nd       = init_parameter.at['lr_rudder_nd','params_init']
        self.kx_rudder          = init_parameter.at['kx_rudder','params_init']
        self.kx_rudder_reverse  = init_parameter.at['kx_rudder_reverse','params_init']
        self.epsilon_rudder     = init_parameter.at['epsilon_rudder','params_init']
        self.cpr_rudder         = init_parameter.at['cpr_rudder','params_init'] 
        self.gammaN             = init_parameter.at['gammaN_rudder','params_init']
        self.gammaP             = init_parameter.at['gammaP_rudder','params_init']
        self.coeff_urudder_zero = init_parameter.at['coeff_urudder_zero','params_init']
        ### switch computation model
        self.switch_prop_x     = model_switch.at['switch_prop_x','value'] # 1: Hachii 2: Yasukawa for thrust computation on 2,3,4th q.
        self.switch_prop_y     = model_switch.at['switch_prop_y','value'] # 0: YP=0 at n>0, 1: use exp.polynomial for 2nd q 
        self.switch_prop_n     = model_switch.at['switch_prop_n','value'] # 0: NP=0 at n>0, 1: use exp.polynomial for 2nd q
        self.switch_rudder     = model_switch.at['switch_rudder','value'] # 0: YR & NR are 0 at n<0, 1: compute force at n<0
        self.switch_fn_rudder  = model_switch.at['switch_fn','value']     # model for rudder normal force, 1: Fujii, 2: lindenburg
        self.switch_ur_rudder  = model_switch.at['switch_ur','value']     # model for effective inflow angle of rudder, 1: standard MMG by Yoshimura, 2: reversed model by Yasukawa on u<0
        self.switch_wind       = model_switch.at['switch_wind','value'] # 0: off, 1: on
        self.switch_windtype   = model_switch.at['switch_windtype','value'] # 1: uniform wind, 2: irregular wind, 3: instantaneous wind measured by exp (converted to true), 4: use exp relavite wind
        
        # ### wind force coefficients
        if 'XX0' in init_parameter.index :
            self.XX0 = init_parameter.at['XX0','params_init'] 
            self.XX1 = init_parameter.at['XX1','params_init'] 
            self.XX3 = init_parameter.at['XX3','params_init'] 
            self.XX5 = init_parameter.at['XX5','params_init'] 
            self.YY1 = init_parameter.at['YY1','params_init'] 
            self.YY3 = init_parameter.at['YY3','params_init']
            self.YY5 = init_parameter.at['YY5','params_init']
            self.NN1 = init_parameter.at['NN1','params_init']  
            self.NN2 = init_parameter.at['NN2','params_init']  
            self.NN3 = init_parameter.at['NN3','params_init']
            self.KK1 = 0
            self.KK2 = 0
            self.KK3 = 0
            self.KK5 = 0  

        else:
            self.windcoeff =wind.windCoeff(self.lpp, self.area_projected_trans, self.area_projected_lateral, self.area_projected_lateral_superstructure,
                                            self.lcw, self.lcbr, self.hbr, self.hc_AL, self.breadth_wind, self.swayforce_to_cg)
            self.XX0 = self.windcoeff[0]
            self.XX1 = self.windcoeff[1]
            self.XX3 = self.windcoeff[2]
            self.XX5 = self.windcoeff[3]
            self.YY1 = self.windcoeff[4]
            self.YY3 = self.windcoeff[5]
            self.YY5 = self.windcoeff[6]
            self.NN1 = self.windcoeff[7]
            self.NN2 = self.windcoeff[8]
            self.NN3 = self.windcoeff[9]
            self.KK1 = self.windcoeff[10]
            self.KK2 = self.windcoeff[11]
            self.KK3 = self.windcoeff[12]
            self.KK5 = self.windcoeff[13]

    def setParams(self, parameter):
        """ update parameters (hydro derivatives etc) of MMG model
        
        Args:
            parameter(dataflame, float): set of parameters to update
        
        Returns:
            none
        """
        self.MassX_nd      = parameter.at['massx_nd','value'] 
        self.MassY_nd      = parameter.at['massy_nd','value'] 
        self.IJzz_nd       = parameter.at['IzzJzz_nd','value'] 
        # Hull
        self.Xuu_nd        = parameter.at['xuu_nd','value'] 
        self.Xvr_nd        = parameter.at['xvr_nd','value'] 
        self.Yv_nd         = parameter.at['yv_nd','value'] 
        self.Yr_nd         = parameter.at['yr_nd','value']
        self.Nv_nd         = parameter.at['nv_nd','value'] 
        self.Nr_nd         = parameter.at['nr_nd','value'] 
        self.CD            = parameter.at['coeff_drag_sway','value']#0.500
        self.C_rY          = parameter.at['cry_cross_flow','value'] #1.00
        self.C_rN          = parameter.at['crn_cross_flow','value'] #0.50
        self.X_0F_nd       = self.Xuu_nd
        self.X_0A_nd       = parameter.at['coeff_drag_aft','value']
        # Propeller
        self.t_prop             = parameter.at['t_prop','value']  
        self.wP0           = parameter.at['w_prop_zero','value']  
        self.tau           = parameter.at['tau_prop','value'] 
        self.CP_nd         = parameter.at['coeff_cp_prop','value'] 
        self.xP_nd         = parameter.at['xp_prop_nd','value'] 
        self.kt_coeff      = np.array([parameter.at['kt_coeff0','value'], parameter.at['kt_coeff1','value'], 
                                        parameter.at['kt_coeff2','value']]) 
        self.Ai            = np.array([parameter.at['ai_coeff_prop0','value'], parameter.at['ai_coeff_prop1','value'], 
                                        parameter.at['ai_coeff_prop2','value'], parameter.at['ai_coeff_prop3','value'],
                                        parameter.at['ai_coeff_prop4','value'], parameter.at['ai_coeff_prop5','value'],
                                        parameter.at['ai_coeff_prop6','value'], parameter.at['ai_coeff_prop7','value']]) 
        self.Bi            = np.array([parameter.at['bi_coeff_prop0','value'], parameter.at['bi_coeff_prop1','value'], 
                                        parameter.at['bi_coeff_prop2','value'], parameter.at['bi_coeff_prop3','value'],
                                        parameter.at['bi_coeff_prop4','value'], parameter.at['bi_coeff_prop5','value'],
                                        parameter.at['bi_coeff_prop6','value'], parameter.at['bi_coeff_prop7','value']])
        self.Ci            = np.array([parameter.at['ci_coeff_prop0','value'], parameter.at['ci_coeff_prop1','value'], 
                                        parameter.at['ci_coeff_prop2','value'], parameter.at['ci_coeff_prop3','value']]) 
        self.Jmin          = parameter.at['Jmin','value']  # -0.5 coeff. from exp
        self.alpha_p       = parameter.at['alpha_prop','value']  # 1.5  coeff. from exp   
        
        # Rudder
        self.t_rudder           = parameter.at['t_rudder','value'] 
        self.ah_rudder          = parameter.at['ah_rudder','value']
        self.xh_rudder_nd       = parameter.at['xh_rudder_nd','value'] 
        self.lr_rudder_nd       = parameter.at['lr_rudder_nd','value']
        self.kx_rudder          = parameter.at['kx_rudder','value']
        self.kx_rudder_reverse  = parameter.at['kx_rudder_reverse','value']
        self.epsilon_rudder     = parameter.at['epsilon_rudder','value']
        self.cpr_rudder         = parameter.at['cpr_rudder','value'] 
        self.gammaN             = parameter.at['gammaN_rudder','value']
        self.gammaP             = parameter.at['gammaP_rudder','value']
        
    ##### Equation of motion #####
    def hydroForceLs(self, physical_time, x, n_prop, delta_rudder, wind_velo_true, wind_dir_true,save_intermid):
        """ return accel. of ship by mmg model for low speed
        
        return accelerations of ship by mmg model for low speed from state value (position, speed, heading) and control (propeller rev., rudder angle).
        process several timestep's data at once.(use np.ndarray)
        Args:
            physical_time (ndarray, float) : physical time of sim (for irregular wind comp.) shape =[number_of_data, 1]
            x (ndarray, float) : set of state values. comtain data for several timestep (small batch).
                                x.shape = [number_of_timestep, number_of_stateval]
                                x = [x_position, u_velo, y_position, vm_velo, psi_hat, r_angvelo]
            n_prop (ndarray, float) : set of propeller revolution. (rps). n_prop.shape = [number_of_timestep]
            delta_rudder(ndarray, float): set of angle of rudder (!! rad !!). [-pi, pi], delta_rudder.shape = [number_of_timestep]
            wind_velo_true (ndarray, float) : true wind velocity (velocity to ground.) if don't calculate wind force, put  np.zeros.
                                if use numerical wind (uniform of irregular), put average true wind for all timestep.
                                if use experimental result measured by Anemoeter, put instant value converted to midship from experimental result.
                                unit = (m/s). shape = [number_of_timestep]
            wind_dir_true (ndarray, float) : true wind direction (direction to ground). related to the earth fix coordinate.
                                            !! please convert to earth-fix coordinate x-direction you are using, in some case, north is not zero.  
                                            treat as same as wind_velo_true.
                                            unit = (rad), [0, 2pi]. shape = [number_of_timestep], clock-wise.
        Retruns:
            retrun (float, ndarray): time derivative of state value driven by MMG model. return.shape= [number_of_timestep, number_of_stateval]
                                    [X_dot, u_dot, Y_dot, vm_dot, psi_dot(=r_angvelo), r_dot]
        
        Note:
            Detail of state values:
            x_position_mid : mid ship position of ship on earth-fix coordinate: x_hat - y_hat system(= X-Y system).
            u_velo : ship fix velocity of ship heading direction.
            y_position_mid : mid ship potision of ship on earth-fix coordinate: x_hat - y_hat system(= X-Y system).
            vm_velo: ship fix velocity of ship breadth direction.
            psi_hat : heading of ship. psi_hat = 0 at x_hat direction.
            r_angvelo: angular velocity of ship heading (Yaw).
            
            Detail of time derivative of state values:
            x_dot : time derivative of x_position_mid. Earth fix velocity of ship, drive from coor. conversion of u_velo.
            u_dot : time derivative of u_velo. ship fix acceleration of ship heading direction.
            y_dot : time derivative of y_position_mid. Earth fix velocity of ship, drive from coor. conversion of vm_velo.
            u_dot : time derivative of vm_velo. ship fix acceleration of ship breadth direction.
            psi_dot: time derivative of psi_hat. which equal to r_angvelo.
            r_dot : time derivative of r_angvelo.            
        """        
        ### read State Variables ###
        x_position_mid = np.copy(x[:, 0:1])
        u_velo       = np.copy(x[:, 1:2])
        y_position_mid = np.copy(x[:, 2:3])
        vm_velo      = np.copy(x[:, 3:4])
        psi_hat      = np.copy(x[:, 4:5]) # psi_hat obtained by MMG (!! rad !!)
        r_angvelo    = np.copy(x[:, 5:6])
        
        ### main ###
        # Forward velocity
        U_ship    = np.sqrt(u_velo**2 + vm_velo**2)
        beta_hat = np.arctan2(vm_velo, u_velo) # = -beta
        
        # add dimension
        Dim_add_M  = 0.5 * self.rho_fresh * self.lpp**2 * self.draft
        Dim_add_I  = 0.5 * self.rho_fresh * self.lpp**4 * self.draft
        Dim_add_uv = 0.5 * self.rho_fresh * self.lpp    * self.draft * U_ship**2
        Dim_add_r  = 0.5 * self.rho_fresh * self.lpp**2 * self.draft * U_ship**2
        Mass  = self.Mass_nd  * Dim_add_M
        MassX = self.MassX_nd * Dim_add_M
        MassY = self.MassY_nd * Dim_add_M
        IJzz  = self.IJzz_nd  * Dim_add_I
        xG    = self.xG_nd    * self.lpp

        # avoid zero divide at approx. U_ship = 0
        u_nd = np.where(np.abs(U_ship) < 1.0e-5, 1.0e-7 * np.ones(u_velo.shape),  u_velo  / U_ship        ) 
        v_nd = np.where(np.abs(U_ship) < 1.0e-5, 1.0e-7 * np.ones(vm_velo.shape), vm_velo / U_ship        )
        r_nd = np.where(np.abs(U_ship) < 1.0e-5, 1.0e-7 * np.ones(r_angvelo.shape),  r_angvelo * self.lpp / U_ship)
        U_ship         = np.where(np.abs(U_ship) < 1.0e-5, 1.0e-7 * np.ones(U_ship.shape),  U_ship             )
        
        ##########################
        ### Force of Hull ###
        ##########################
        hull_force = hullforce.hullForceYoshimuraSimpson(U_ship, u_velo, vm_velo, r_angvelo, beta_hat, self.lpp, self.draft, self.rho_fresh, self.X_0F_nd, self.X_0A_nd,
                                                    self.Xvr_nd, self.Yv_nd, self.Yr_nd, self.Nv_nd, self.Nr_nd, self.C_rY, self.C_rN, self.CD)
        # hull_force = hullforce.hullForceYoshimura(U_ship, u_velo, vm_velo, r_angvelo, beta_hat, self.lpp, self.draft, self.rho_fresh, self.X_0F_nd, self.X_0A_nd,
                                                    # self.Xvr_nd, self.Yv_nd, self.Yr_nd, self.Nv_nd, self.Nr_nd, self.C_rY, self.C_rN, self.CD)
        XH = hull_force[:, 0:1]
        YH = hull_force[:, 1:2]
        NH = hull_force[:, 2:3]
        
        ##########################
        ### Force of Propeller ###
        ##########################
        prop_force=  propellerforce.get_propellerforce(u_velo, vm_velo, r_angvelo, n_prop, 
                    self.rho_fresh, self.lpp, self.draft, self.dia_prop, self.pitchr,  
                    self.t_prop, self.wP0, self.tau, self.CP_nd, self.xP_nd, self.kt_coeff, 
                    self.Ai, self.Bi, self.Ci, self.Jmin, self.alpha_p, 
                    self.switch_prop_x, self.switch_prop_y, self.switch_prop_n)
        XP = prop_force[:, 0:1]
        YP = prop_force[:, 1:2]
        NP = prop_force[:, 2:3]
        one_minus_wprop = prop_force[:, 3:4]
        J_prop = prop_force[:, 4:5]
        ##########################
        ### Force of Ruder ###
        ##########################
        rudder_force = rudderforce.get_rudderforce(u_velo, vm_velo, r_angvelo, delta_rudder, n_prop,
                    one_minus_wprop, self.t_prop, J_prop,
                    self.rho_fresh, self.lpp, self.dia_prop, self.area_rudder, self.lambda_rudder, self.x_location_rudder , 
                    self.t_rudder, self.ah_rudder, self.xh_rudder_nd, self.lr_rudder_nd, self.kx_rudder, self.epsilon_rudder,
                    self.gammaP, self.gammaN,
                    self.kx_rudder_reverse, self.cpr_rudder, self.coeff_urudder_zero,
                    XP, self.switch_rudder, self.switch_fn_rudder, self.switch_ur_rudder)
        XR = rudder_force[:, 0:1]
        YR = rudder_force[:, 1:2]
        NR = rudder_force[:, 2:3]
        aR = rudder_force[:, 3:4]
        resultant_U_rudder = rudder_force[:, 4:5]
        u_rudder = rudder_force[:, 5:6]
        v_rudder = rudder_force[:, 6:7]
        
        ##########################
        ### Force of wind ###
        ##########################
        XA = np.zeros((x_position_mid.size, 1))
        YA = np.zeros((x_position_mid.size, 1))
        NA = np.zeros((x_position_mid.size, 1))
        if self.switch_wind == 0: # no wind force
            pass
        ### compute instant relative(apparent) wind dirction and speed
        elif self.switch_wind == 1:
            
            if self.switch_windtype == 1: #uniform
                wind_velo_true_instant = np.copy(wind_velo_true)
                wind_dir_true_instant  = np.copy(wind_dir_true)
                ### compute relative wind on midship
                wind_relative = relative.true2Apparent(wind_velo_true_instant, U_ship, wind_dir_true_instant, beta_hat, psi_hat)
                wind_velo_relative = np.copy(wind_relative[:,0:1])
                angle_of_attack = 2.0*np.pi-np.copy(wind_relative[:,1:2])    
            elif self.switch_windtype == 2: # irregular wind
                wind_velo_true_instant = wind.irregularWind(wind_velo_true, physical_time)
                wind_dir_true_instant  = np.copy(wind_dir_true)
                ### compute relative wind on midship
                wind_relative = relative.true2Apparent(wind_velo_true_instant, U_ship, wind_dir_true_instant, beta_hat, psi_hat)
                wind_velo_relative = np.copy(wind_relative[:,0:1])
                angle_of_attack = 2.0*np.pi-np.copy(wind_relative[:,1:2])
            elif self.switch_windtype == 3: # measured wind by anemometer on model ship
                ### !!!! caution !!!!!
                ### conevert to true wind(wind_velo_true, wind_dir_true) from anemometer output 
                ### before use this part
                wind_velo_true_instant = np.copy(wind_velo_true)
                wind_dir_true_instant  = np.copy(wind_dir_true)
                ### compute relative wind on midship
                wind_relative = relative.true2Apparent(wind_velo_true_instant, U_ship, wind_dir_true_instant, beta_hat, psi_hat)
                wind_velo_relative = np.copy(wind_relative[:,0:1])
                angle_of_attack = 2.0*np.pi-np.copy(wind_relative[:,1:2])
            elif self.switch_windtype == 4: # use apparent wind on model, not recommend to use, for debug
                wind_velo_relative = np.copy(wind_velo_true)
                angle_of_attack = 2.0*np.pi-np.copy(wind_dir_true)
        
            ### compute wind force by relative wind 
            wind_force = wind.windForce(wind_velo_relative, angle_of_attack, self.area_projected_trans,
                                    self.area_projected_lateral, self.swayforce_to_cg,
                                    self.rho_air, self.lpp, self.XX0, self.XX1, self.XX3, self.XX5,
                                    self.YY1, self.YY3, self.YY5, self.NN1, self.NN2, self.NN3,
                                    self.KK1, self.KK2, self.KK3, self.KK5)
            XA = wind_force[:, 0:1]
            YA = wind_force[:, 1:2]
            NA = wind_force[:, 2:3]
        
        ######################
        ###    Summation of every force and moment
        ######################
        X = XH + XP + XR + XA
        Y = YH + YP + YR + YA
        N = NH + NP + NR + NA

        AA1 = Mass + MassY
        AA2 = xG * Mass
        AA3 = Y - (Mass+MassX) * u_velo * r_angvelo
        BB1 = IJzz + xG**2 * Mass
        BB2 = xG * Mass
        BB3 = N - xG * Mass * u_velo * r_angvelo

        u_dot  = (X + (Mass+MassY)*vm_velo*r_angvelo 
                + xG*Mass*r_angvelo**2) / (Mass+MassX)
        vm_dot = (AA3*BB1 - AA2*BB3) / (AA1*BB1 - AA2*BB2)
        r_dot  = (AA3*AA2 - BB3*AA1)/ (AA2*BB2 - AA1*BB1)
        x_dot = u_velo * np.cos(psi_hat) - vm_velo * np.sin(psi_hat) 
        y_dot = u_velo * np.sin(psi_hat) + vm_velo * np.cos(psi_hat)

        #### set intermideate variables as attribute for postprocess####
        if(save_intermid):
            self.XH = XH
            self.YH = YH
            self.NH = NH

            self.XP = XP
            self.YP = YP
            self.NP = NP

            self.XR = XR
            self.YR = YR
            self.NR = NR

            self.XA = XA
            self.YA = YA
            self.NA = NA

            self.aR = aR
            self.resultant_U_rudder = resultant_U_rudder
            self.u_rudder = u_rudder
            self.v_rudder = v_rudder
            
        return np.concatenate([x_dot, u_dot, y_dot, vm_dot, r_angvelo, r_dot], axis=1)