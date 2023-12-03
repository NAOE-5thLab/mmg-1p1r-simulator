from subroutine import MMG
from subroutine import runge_kutta as rk
from subroutine import trajectory as traj
from subroutine import relative
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

### debug 
# init

# read principal particulars from csv6
principal_particulars = pd.read_csv('./Hydro/MMG/input_csv/principal_particulars.csv', header=0, index_col=0)
# read parameters from csv
parameter_init = pd.read_csv('./Hydro/MMG/input_csv/MMG_params.csv', header=0, index_col=0)

# read model switch param form csv
switch = pd.read_csv('./Hydro/MMG/input_csv/model_switch.csv',header=0, index_col=0)


# plot params
plt.rcParams["font.family"] = "serif"       # 使用するフォント
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 16              # 基本となるフォントの大きさ
plt.rcParams["mathtext.cal"] = "serif"      # TeX表記に関するフォント設定
plt.rcParams["mathtext.rm"] = "serif"       # TeX表記に関するフォント設定
plt.rcParams["mathtext.it"] = "serif:italic"# TeX表記に関するフォント設定
plt.rcParams["mathtext.bf"] = "serif:bold"  # TeX表記に関するフォント設定
plt.rcParams["mathtext.fontset"] = "cm"     # TeX表記に関するフォント設定

# read experiment data
exp_data = pd.read_csv('./Hydro/MMG/input_csv/stopping_processed_CRASH_ASTERN_1200-1200_-35.xlsx.csv', header=0, index_col=0) 
dt_sec = exp_data.index[2:3].values - exp_data.index[1:2].values
# instantiate mmg model
mmg2 = MMG.MmgModel(principal_particulars, parameter_init, switch)
lpp = mmg2.lpp
breadth = mmg2.B

# yasukawa's model
mmg2.switch_windtype = 3

# set start time step of MMG simulation
startstep = 0  
# set control variable
n_prop = exp_data.loc[exp_data.index[startstep:],'n_prop [rps]'].values
delta_rudder = exp_data.loc[exp_data.index[startstep:],'delta_rudder [rad]'].values
no_timestep=len(n_prop)
# set initial sate value 
stateval  = np.empty((no_timestep, 6)) #[x_pos, u, y_pos, vm, psi, r]
stateval[0:1, 0:1] = exp_data.at[exp_data.index[startstep], 'x_position_mid [m]'] #x_pos t=0
stateval[0:1, 1:2] = exp_data.at[exp_data.index[startstep], 'u_velo [m/s]'] #u t=0
stateval[0:1, 2:3] = exp_data.at[exp_data.index[startstep], 'y_position_mid [m]'] #y_pos t=0
stateval[0:1, 3:4] = exp_data.at[exp_data.index[startstep], 'vm_velo [m/s]'] #vm t=0
stateval[0:1, 4:5] = exp_data.at[exp_data.index[startstep], 'psi_hat [rad]']#psi t=0
stateval[0:1, 5:6] = exp_data.at[exp_data.index[startstep], 'r_angvelo [rad/s]'] #r t=0
# set true wind (from experiment result, USE TURE WIND!!) 
windv  = exp_data.loc[exp_data.index[startstep:],'wind_velo_true [m/s]'].values
windd  = exp_data.loc[exp_data.index[startstep:],'wind_dir_true [rad]'].values
windv = windv[:, np.newaxis]
windd = windd[:, np.newaxis]
print(windv.shape)


# compute MMG model by RK4
start = time.time()
for i in range(no_timestep-1):
    slope = rk.rk4_mmg(dt_sec, mmg2.hydroForceLs, exp_data.index[startstep+i:startstep+i+1].values, stateval[i:i+1,:], n_prop[i:i+1], 
                        delta_rudder[i:i+1], windv[i:i+1], windd[i:i+1])
    stateval[i+1:i+2, :] = stateval[i:i+1, :] + slope[:] * dt_sec
    #  physical_time[i+1,] = (i+1)*dt_sec
    if (i % int(100) == 0):
        print("step : ", i)
elapsed_time = time.time()-start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")    
# save results
valid_yasukawa = np.copy(stateval)

# hachii's model
mmg2.switch_prop_x =1

# set initial sate value 
stateval  = np.empty((no_timestep, 6)) #[x_pos, u, y_pos, vm, psi, r]
stateval[0:1, 0:1] = exp_data.at[exp_data.index[startstep], 'x_position_mid [m]'] #x_pos t=0
stateval[0:1, 1:2] = exp_data.at[exp_data.index[startstep], 'u_velo [m/s]'] #u t=0
stateval[0:1, 2:3] = exp_data.at[exp_data.index[startstep], 'y_position_mid [m]'] #y_pos t=0
stateval[0:1, 3:4] = exp_data.at[exp_data.index[startstep], 'vm_velo [m/s]'] #vm t=0
stateval[0:1, 4:5] = exp_data.at[exp_data.index[startstep], 'psi_hat [rad]']#psi t=0
stateval[0:1, 5:6] = exp_data.at[exp_data.index[startstep], 'r_angvelo [rad/s]'] #r t=0


start = time.time()
for i in range(no_timestep-1):
    slope = rk.rk4_mmg(dt_sec, mmg2.hydroForceLs, exp_data.index[startstep+i:startstep+i+1].values, stateval[i:i+1,:], n_prop[i:i+1], 
                        delta_rudder[i:i+1], windv[i:i+1], windd[i:i+1])
    stateval[i+1:i+2, :] = stateval[i:i+1, :] + slope[:] * dt_sec
    #  physical_time[i+1,] = (i+1)*dt_sec
    if (i % int(100) == 0):
        print("step : ", i)
elapsed_time = time.time()-start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")     

valid_hachii = np.copy(stateval)

#####
# with out wind
#####
# yasukawa's model
mmg2.switch_prop_x =2
mmg2.switch_prop_y =1
mmg2.switch_prop_n =1
mmg2.switch_wind = 0 # 0: wind force computation off
mmg2.switch_windtype =3

# set initial sate value 
stateval  = np.empty((no_timestep, 6)) #[x_pos, u, y_pos, vm, psi, r]
stateval[0:1, 0:1] = exp_data.at[exp_data.index[startstep], 'x_position_mid [m]'] #x_pos t=0
stateval[0:1, 1:2] = exp_data.at[exp_data.index[startstep], 'u_velo [m/s]'] #u t=0
stateval[0:1, 2:3] = exp_data.at[exp_data.index[startstep], 'y_position_mid [m]'] #y_pos t=0
stateval[0:1, 3:4] = exp_data.at[exp_data.index[startstep], 'vm_velo [m/s]'] #vm t=0
stateval[0:1, 4:5] = exp_data.at[exp_data.index[startstep], 'psi_hat [rad]']#psi t=0
stateval[0:1, 5:6] = exp_data.at[exp_data.index[startstep], 'r_angvelo [rad/s]'] #r t=0

start = time.time()
for i in range(no_timestep-1):
    slope = rk.rk4_mmg(dt_sec, mmg2.hydroForceLs, exp_data.index[startstep+i:startstep+i+1].values, stateval[i:i+1,:], n_prop[i:i+1], 
                        delta_rudder[i:i+1], windv[i:i+1], windd[i:i+1])
    stateval[i+1:i+2, :] = stateval[i:i+1, :] + slope[:] * dt_sec
    #  physical_time[i+1,] = (i+1)*dt_sec
    if (i % int(100) == 0):
        print("step : ", i)
elapsed_time = time.time()-start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")      

valid_yasukawa_nowind = np.copy(stateval)

# hachii's model
mmg2.switch_prop_x =1


stateval  = np.empty((no_timestep, 6)) #[x_pos, u, y_pos, vm, psi, r]
stateval[0:1, 0:1] = exp_data.at[exp_data.index[startstep], 'x_position_mid [m]'] #x_pos t=0
stateval[0:1, 1:2] = exp_data.at[exp_data.index[startstep], 'u_velo [m/s]'] #u t=0
stateval[0:1, 2:3] = exp_data.at[exp_data.index[startstep], 'y_position_mid [m]'] #y_pos t=0
stateval[0:1, 3:4] = exp_data.at[exp_data.index[startstep], 'vm_velo [m/s]'] #vm t=0
stateval[0:1, 4:5] = exp_data.at[exp_data.index[startstep], 'psi_hat [rad]']#psi t=0
stateval[0:1, 5:6] = exp_data.at[exp_data.index[startstep], 'r_angvelo [rad/s]'] #r t=0

start = time.time()
for i in range(no_timestep-1):
    slope = rk.rk4_mmg(dt_sec, mmg2.hydroForceLs, exp_data.index[startstep+i:startstep+i+1].values, stateval[i:i+1,:], n_prop[i:i+1], 
                        delta_rudder[i:i+1], windv[i:i+1], windd[i:i+1])
    stateval[i+1:i+2, :] = stateval[i:i+1, :] + slope[:] * dt_sec
    #  physical_time[i+1,] = (i+1)*dt_sec
    if (i % int(100) == 0):
        print("step : ", i)
elapsed_time = time.time()-start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")    

valid_hachii_nowind = np.copy(stateval)

#mkplot
## plot tajectory
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_ylabel("$\hat{x} [m]$")
ax1.set_xlabel("$\hat{y} [m]$")
ax1.plot(exp_data.loc[:,'y_position_mid [m]'], exp_data.loc[:,'x_position_mid [m]'], color="black", linewidth = 0.7, label="exp")
# ax1.plot(valid_yasukawa[:, 2:3], valid_yasukawa[:, 0:1], color="#0277BD", linewidth = 0.7, label="Yasukawa's model for $X_{p}$")
ax1.plot(valid_hachii[:, 2:3], valid_hachii[:, 0:1], color="navy", linewidth = 0.7, label="Hachii's model for $X_{p}$")
# ax1.plot(valid_yasukawa_nowind[:, 2:3], valid_yasukawa_nowind[:, 0:1], color="#0277BD", linewidth = 0.7, linestyle = '--', label="Yasukawa's model for $X_{p},$ w/o wind")
ax1.plot(valid_hachii_nowind[:, 2:3], valid_hachii_nowind[:, 0:1], color="tomato", linewidth = 0.7, linestyle = '--', label="Hachii's model for $X_{p},$ w/o wind")
for i, ti in enumerate(exp_data.index.values):
# for i in range(1):
    if ti % 5 == 0:
        # plot exp
        x1,x2,x3,x4,x5 = traj.ship_coo(
        np.array([exp_data.at[exp_data.index[i], 'x_position_mid [m]'], exp_data.at[exp_data.index[i], 'y_position_mid [m]'], exp_data.at[exp_data.index[i], 'psi_hat [rad]']]), lpp, breadth, scale=1)
        poly = plt.Polygon((x1[::-1],x2[::-1],x3[::-1],x4[::-1],x5[::-1]), fc="w", alpha=0.5, linewidth = 0.5, ec="black")
        ax1.add_patch(poly)

for i, ti in enumerate(exp_data.index[startstep:].values):
# for i in range(1):
    if ti % 10 == 0:        
        # plot exp wind relative
        exprwindend = np.array([exp_data.at[exp_data.index[i], 'y_position_mid [m]'], exp_data.at[exp_data.index[i], 'x_position_mid [m]']])
        exprwindstart = np.copy(exprwindend)
        exprwindstart[0,]   = exprwindstart[0] + exp_data.at[exp_data.index[i], 'wind_velo_relative_mid [m/s]'] *np.sin(exp_data.at[exp_data.index[i], 'wind_dir_relative_mid [rad]']+exp_data.at[exp_data.index[i], 'psi_hat [rad]'])
        exprwindstart[1,]   = exprwindstart[1] + exp_data.at[exp_data.index[i], 'wind_velo_relative_mid [m/s]'] *np.cos(exp_data.at[exp_data.index[i], 'wind_dir_relative_mid [rad]']+exp_data.at[exp_data.index[i], 'psi_hat [rad]'],)

        ax1.annotate(s='',xy=exprwindend,xytext=exprwindstart,arrowprops=dict(facecolor='gray', edgecolor='gray', width =1.0,headwidth=7.0,headlength=7.0,shrink=0.01))

        # # plot yasukawa's model
        # yasukawax1, yasukawax2, yasukawax3, yasukawax4, yasukawax5 = traj.ship_coo(
        # np.array([valid_yasukawa[i, 0], valid_yasukawa[i, 2], valid_yasukawa[i, 4]]), lpp, breadth, scale=1)
        # yasukawapoly = plt.Polygon((yasukawax1[::-1], yasukawax2[::-1], yasukawax3[::-1], yasukawax4[::-1], yasukawax5[::-1]), fc="w", alpha=0.5, linewidth = 0.5, ec="#0277BD")
        # ax1.add_patch(yasukawapoly)
        # # ## relative wind
        # U_ship = np.sqrt(valid_yasukawa[i:i+1,1:2]**2+valid_yasukawa[i:i+1,3:4]**2)
        # beta = np.arctan2(valid_yasukawa[i:i+1,3:4], valid_yasukawa[i:i+1,1:2])
        # relativew = relative.true2Apparent(windv[i:i+1], U_ship, windd[i:i+1],beta, valid_yasukawa[i,4:5] )
        # rwindnorm = 1.0 * relativew[:,0:1]
        # rwindend = np.array([valid_yasukawa[i, 2], valid_yasukawa[i, 0]])
        # rwindstart = np.copy(rwindend)
        # rwindstart[0,]   = rwindstart[0] + rwindnorm *np.sin(valid_yasukawa[i, 4]+relativew[:,1:2])
        # rwindstart[1,]   = rwindstart[1] + rwindnorm *np.cos(valid_yasukawa[i, 4]+relativew[:,1:2])
        # ax1.annotate(s='',xy=rwindend,xytext=rwindstart,xycoords='data',\
        #     arrowprops=dict(facecolor='#0277BD', edgecolor='#0277BD', width =1.0,alpha=0.5,headwidth=7.0,headlength=7.0,shrink=0.01))
        
        # # plot hachii's model
        hachiix1, hachiix2, hachiix3, hachiix4, hachiix5 = traj.ship_coo(
        np.array([valid_hachii[i, 0], valid_hachii[i, 2], valid_hachii[i, 4]]), lpp, breadth, scale=1)
        hachiipoly = plt.Polygon((hachiix1[::-1], hachiix2[::-1], hachiix3[::-1], hachiix4[::-1], hachiix5[::-1]), fc="w", alpha=0.5, linewidth = 0.5, ec="navy")
        ax1.add_patch(hachiipoly)
        
        # # # plot yasukawa's model - nowind 
        # yasukawa_nowindx1, yasukawa_nowindx2, yasukawa_nowindx3, yasukawa_nowindx4, yasukawa_nowindx5 = traj.ship_coo(
        # np.array([valid_yasukawa_nowind[i, 0], valid_yasukawa_nowind[i, 2], valid_yasukawa_nowind[i, 4]]), lpp, breadth, scale=1)
        # yasukawa_nowindpoly = plt.Polygon((yasukawa_nowindx1[::-1], yasukawa_nowindx2[::-1], yasukawa_nowindx3[::-1], yasukawa_nowindx4[::-1], yasukawa_nowindx5[::-1]), fc="w", alpha=0.5, linewidth = 0.5, ec="#0277BD")
        # ax1.add_patch(yasukawa_nowindpoly)

        # # plot yasukawa's model - nowind 
        hachii_nowindx1, hachii_nowindx2, hachii_nowindx3, hachii_nowindx4, hachii_nowindx5 = traj.ship_coo(
        np.array([valid_hachii_nowind[i, 0], valid_hachii_nowind[i, 2], valid_hachii_nowind[i, 4]]), lpp, breadth, scale=1)
        hachii_nowindpoly = plt.Polygon((hachii_nowindx1[::-1], hachii_nowindx2[::-1], hachii_nowindx3[::-1], hachii_nowindx4[::-1], hachii_nowindx5[::-1]), fc="w", alpha=0.5, linewidth = 0.5, ec="tomato")
        ax1.add_patch(hachii_nowindpoly)

ax1.set_ylim(-5, 35)
ax1.set_xlim(-20, 10)
ax1.legend(loc='lower left')
ax1.set_aspect('equal')

fig2 = plt.figure(figsize=(16, 12))
ax1 = fig2.add_subplot(2, 3, 1)
ax1.set_xlabel("$t$ [m/s]")
ax1.set_ylabel("$u$ [m/s]")
ax1.plot(exp_data.index.values, exp_data.loc[:,'u_velo [m/s]'], color="black", linewidth = 0.7, label="exp")
ax1.plot(exp_data.index[startstep:].values, valid_yasukawa[:, 1:2], color="#0277BD", linewidth = 0.7, label="Yasukawa")
ax1.plot(exp_data.index[startstep:].values, valid_hachii[:, 1:2], color="tomato", linewidth = 0.7, label="Hachii")
ax1.plot(exp_data.index[startstep:].values, valid_yasukawa_nowind[:, 1:2], color="#0277BD", linewidth = 0.7, linestyle = '--',label="Yasukawa, w/o wind")
ax1.plot(exp_data.index[startstep:].values, valid_hachii_nowind[:, 1:2], color="tomato", linewidth = 0.7, linestyle = '--',label="Hachii, w/o wind")

#vm
ax2 = fig2.add_subplot(2, 3, 2)
ax2.set_xlabel("$t$ [m/s]")
ax2.set_ylabel("$v_m$ [m/s]")
ax2.plot(exp_data.index.values, exp_data.loc[:,'vm_velo [m/s]'], color="black", linewidth = 0.7, label="exp")
ax2.plot(exp_data.index[startstep:].values, valid_yasukawa[:, 3:4], color="#0277BD", linewidth = 0.7, label="Yasukawa")
ax2.plot(exp_data.index[startstep:].values, valid_hachii[:, 3:4], color="tomato", linewidth = 0.7, label="Hachii")
ax2.plot(exp_data.index[startstep:].values, valid_yasukawa_nowind[:, 3:4], color="#0277BD", linewidth = 0.7, linestyle = '--',label="Yasukawa, w/o wind")
ax2.plot(exp_data.index[startstep:].values, valid_hachii_nowind[:, 3:4], color="tomato", linewidth = 0.7, linestyle = '--',label="Hachii")

#r
ax3 = fig2.add_subplot(2, 3, 3)
ax3.set_xlabel("$t$ [m/s]")
ax3.set_ylabel("$r$ [rad/s]")
ax3.plot(exp_data.index.values, exp_data.loc[:,'r_angvelo [rad/s]'], color="black", linewidth = 0.7, label="exp")
ax3.plot(exp_data.index[startstep:].values, valid_yasukawa[:, 5:6], color="#0277BD", linewidth = 0.7, label="Yasukawa")
ax3.plot(exp_data.index[startstep:].values, valid_hachii[:, 5:6], color="tomato", linewidth = 0.7, label="Hachii")
ax3.plot(exp_data.index.values, valid_yasukawa_nowind[:, 5:6], color="#0277BD", linewidth = 0.7, linestyle = '--',label="Yasukawa, w/o wind")
ax3.plot(exp_data.index.values, valid_hachii_nowind[:, 5:6], color="tomato", linewidth = 0.7, linestyle = '--',label="Hachii")
ax3.grid()
#x
ax4 = fig2.add_subplot(2, 3, 4)
ax4.set_xlabel("$t$ [m/s]")
ax4.set_ylabel("$\hat{x}$ [m/s]")
ax4.plot(exp_data.index.values, exp_data.loc[:,'x_position_mid [m]'], color="black", linewidth = 0.7, label="exp")
ax4.plot(exp_data.index[startstep:].values, valid_yasukawa[:, 0:1], color="#0277BD", linewidth = 0.7, label="Yasukawa")
ax4.plot(exp_data.index[startstep:].values, valid_hachii[:, 0:1], color="tomato", linewidth = 0.7, label="Hachii")
ax4.plot(exp_data.index[startstep:].values, valid_yasukawa_nowind[:, 0:1], color="#0277BD", linewidth = 0.7, linestyle = '--',label="Yasukawa, w/o wind")
ax4.plot(exp_data.index[startstep:].values, valid_hachii_nowind[:, 0:1], color="tomato", linewidth = 0.7, linestyle = '--',label="Hachii, w/o wind")

#y
ax5 = fig2.add_subplot(2, 3, 5)
ax5.set_xlabel("$t$ [m/s]")
ax5.set_ylabel("$\hat{y}$ [m/s]")
ax5.plot(exp_data.index.values, exp_data.loc[:,'y_position_mid [m]'], color="black", linewidth = 0.7, label="exp")
ax5.plot(exp_data.index[startstep:].values, valid_yasukawa[:, 2:3], color="#0277BD", linewidth = 0.7, label="Yasukawa")
ax5.plot(exp_data.index[startstep:].values, valid_hachii[:, 2:3], color="tomato", linewidth = 0.7, label="Hachii")
ax5.plot(exp_data.index[startstep:].values, valid_yasukawa_nowind[:, 2:3], color="#0277BD", linewidth = 0.7,  linestyle = '--',label="Yasukawa")
ax5.plot(exp_data.index[startstep:].values, valid_hachii_nowind[:, 2:3], color="tomato", linewidth = 0.7,  linestyle = '--',label="Hachii")


#psi
ax6 = fig2.add_subplot(2, 3, 6)
ax6.set_xlabel("$t$ [m/s]")
ax6.set_ylabel("$\psi$ [rad]")
ax6.plot(exp_data.index.values, exp_data.loc[:,'psi_hat [rad]'], color="black", linewidth = 0.7, label="exp")
ax6.plot(exp_data.index[startstep:].values, valid_yasukawa[:, 4:5], color="#0277BD", linewidth = 0.7, label="Yasukawa")
ax6.plot(exp_data.index[startstep:].values, valid_hachii[:, 4:5], color="tomato", linewidth = 0.7, label="Hachii")
ax6.plot(exp_data.index[startstep:].values, valid_yasukawa_nowind[:, 4:5], color="#0277BD", linewidth = 0.7, linestyle = '--',label="Yasukawa")
ax6.plot(exp_data.index[startstep:].values, valid_hachii_nowind[:, 4:5], color="tomato", linewidth = 0.7, linestyle = '--',label="Hachii")
fig2.tight_layout()
ax1.legend()
ax4.legend()

plt.show()