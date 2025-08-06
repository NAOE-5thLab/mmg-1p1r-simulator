! MMG model for sigle rudder, single propeller ship "Esso Osaka"
! with artificial Bow thruster and Stern thruster
!
module esso_osaka_vector_input
    implicit none
    
    double precision, parameter :: pi=4.0d0*datan(1.0d0)
    double precision, parameter :: grav=9.80665d0
    double precision, parameter :: rho_fresh=998.0d0/grav

contains

    subroutine MMG_LowSpeed_model(time_derivative, state, delta_rudder, n_prop, n_bt, n_st, Wind_Direction, Wind_velocity,&
        &                         pp_vector, mmg_params_vector)
        implicit none
        
        !! in, out arguments

        double precision, intent(in) :: state(6), delta_rudder, n_prop, n_bt, n_st, Wind_Direction, Wind_velocity
        DOUBLE PRECISION, INTENT(IN) :: pp_vector(24), mmg_params_vector(69)
        double precision, intent(out) :: time_derivative(6)

        !!! local variables
        integer :: i
        integer :: i_max

        double precision :: Mass_nd, MassX_nd, MassY_nd, IzzJzz_nd, xG_nd
        double precision :: u_velo, vm_velo, r_angvelo, psi, Xpos, Ypos
        double precision :: v_nd, r_nd
        double precision :: lpp, breadth, draft, U_ship
        double precision :: D1_ad2, D2_ad2, beta_hat, Mass, MassX, MassY, IJzz, xG
        double precision :: Xuu_nd, Xvr_nd, Yv_nd, Yr_nd, Nv_nd, Nr_nd
        double precision :: CD, C_rY, C_rN
        double precision :: X_0F_nd, X_0A_nd, Y_ad, N_ad
        double precision :: x0, x1, comp0, comp1, comp2, comp3
        double precision :: YHN_nd, NHN_nd
        double precision :: XH, YH, NH
        double precision :: Dp, pitchr, pitch, t_prop, wP0, tau_prop, CP_nd, xP_nd, kt_coeff0, kt_coeff1, kt_coeff2
        double precision :: A1, A2, A3, A4, A5, A6, A7, A8
        double precision :: B1, B2, B3, B4, B5, B6, B7, B8
        double precision :: C3, C6, C7, C10
        double precision :: J_prop, KT, Js, Jsyn, Jsyn0, wP, uprop, one_minus_wprop
        double precision :: XP, YP, NP
        double precision :: AREA, lambda_rudder, hight_rudder, eta_rudder, x_pos_rudder
        double precision :: tR, aH_rudder, xh_rudder_nd, kx_rudder, epsilon_rudder
        double precision :: lR_nd, gammaN, gammaP, kx_rudder_reverse, cpr_rudder, xh_rudder, lR
        double precision :: kappa_rudder, one_minus_wrudder, fa, uR, vR, urpr1, urpr2, ursq, UUR, aR, FN, XR, YR, NR
        double precision :: YBT, NBT, YST, NST
        double precision :: D_bthrust, D_sthrust, x_bowthrust_loc,x_stnthrust_loc, Thruster_speed_max, Fr_number
        double precision :: KT_bow_forward, KT_bow_reverse, aY_bow, aN_bow, KT_stern_forward, KT_stern_reverse, aY_stern, aN_stern 
        double precision :: X, Y, N
        double precision :: u_dot, vm_dot, r_dot, psi_dot, X_dot, Y_dot
        double precision :: TempA1, TempA2, TempA3, TempB1, TempB2, TempB3
    
        !   Valiables for wind forces and moments
        double precision :: XA, YA, NA
        double precision :: Windcoef(10), AT, AL, AOD, LCW, LCBR, HBR, HC, SBW, Lz
        double precision :: Relative_Wind_Direction, Relative_Wind_Velocity, Angle_of_attack
        double precision :: CXwind, CYwind, CNwind, CKwind, FXwind, FYwind, FNwind, FKwind
        double precision :: XX0, XX1, XX3, XX5;
        double precision :: YY1, YY3, YY5;
        double precision :: NN1, NN2, NN3;
        double precision :: KK1, KK2, KK3, KK5
        !
        ! read State Variables and control
        Xpos  = state(1)
        u_velo     = state(2)
        Ypos  = state(3)
        vm_velo    = state(4) ! v at midship (v -->> vm_velo 4/16/2018 Nishikawa)
        psi   = state(5)
        r_angvelo     = state(6)
        
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!!  Principal Particulars and parameters
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! principal particular
        lpp = pp_vector(1)
        breadth = pp_vector(2)
        draft = pp_vector(3)
        Mass_nd  = pp_vector(4)
        MassX_nd = mmg_params_vector(1)
        MassY_nd = mmg_params_vector(2)
        IzzJzz_nd  = mmg_params_vector(3)
        xG_nd    = pp_vector(5)
        ! prop
        Dp    =  pp_vector(6)
        pitchr =  pp_vector(7)
    
        ! Rudder
        AREA    =  pp_vector(8)
        lambda_rudder  =  pp_vector(9)
        x_pos_rudder     = pp_vector(10)

        ! side thrusters
        D_bthrust = pp_vector(11)
        D_sthrust = pp_vector(12)
        x_bowthrust_loc = pp_vector(13)
        x_stnthrust_loc = pp_vector(14)
        Thruster_speed_max = pp_vector(15)
        ! wind force
        AT    = pp_vector(16)  !    AT --- Transverse Projected area (m^2)
        AL    = pp_vector(17)  !    AL --- Lateral Projected area (m^2)
        AOD   = pp_vector(18) !    AOD -- Lateral Projected area of superstructure Ass and LNG tanks,
                            !           container etc. on the deck (m^2)
                            !           Here, Ass is defined as lateral Projected area of superstructure (m^2)
        LCW   = pp_vector(19) !    C ---- Distance from midship section to center of lateral projected area (m)
        LCBR  = pp_vector(20) !    CBR -- Distance from midship section to center of the Ass (m)
        HBR   = pp_vector(21)    !    HBR -- Height from free surface to top of the superstructure (bridge) (m) 
        HC    = pp_vector(22)  !    HC --- Height to center of lateral projected area (m)
        SBW   = pp_vector(23)         !    SBW -- breadth for wind area (in usual, it coincides with breadth, but it is defined for Trimaran vessel)
        Lz    = pp_vector(24)     !    Lz --- Acting position of sway force from center of gravity (it is necessaty to calculate roll moment)
    
        !!! parameters !!!
        ! Hull force parameters    
        Xuu_nd = mmg_params_vector(4)
        Xvr_nd = mmg_params_vector(5)
        Yv_nd  = mmg_params_vector(6)
        Yr_nd  = mmg_params_vector(7)
        Nv_nd  = mmg_params_vector(8)
        Nr_nd  = mmg_params_vector(9)

        CD   = mmg_params_vector(10)
        C_rY = mmg_params_vector(11)
        C_rN = mmg_params_vector(12)


        X_0F_nd = Xuu_nd
        X_0A_nd = mmg_params_vector(13)
        
        ! Propeller
        t_prop     = mmg_params_vector(14)
        wP0        = mmg_params_vector(15)
        tau_prop   = mmg_params_vector(16)
        CP_nd      = mmg_params_vector(17)
        xP_nd      = mmg_params_vector(18)
        kt_coeff0  = mmg_params_vector(19)
        kt_coeff1  = mmg_params_vector(20)
        kt_coeff2  = mmg_params_vector(21)

        ! coeffs for Yp
        A1 = mmg_params_vector(22)  !A1 to A5: Hachii's paper
        A2 = mmg_params_vector(23)
        A3 = mmg_params_vector(24)
        A4 = mmg_params_vector(25)
        A5 = mmg_params_vector(26)
        A6 = mmg_params_vector(27) !A6 to A8: Seiunmaru's exp. value 
        A7 = mmg_params_vector(28)
        A8 = mmg_params_vector(29)
        ! coeffs for Np  
        B1 = mmg_params_vector(30)   !B1 to B5: Hachii's paper
        B2 = mmg_params_vector(31)
        B3 = mmg_params_vector(32)
        B4 = mmg_params_vector(33)
        B5 = mmg_params_vector(34)
        B6 = mmg_params_vector(35)  !B6 to B8: Seiunmaru's exp. value 
        B7 = mmg_params_vector(36)
        B8 = mmg_params_vector(37)
        ! coeffs for Xp
        C3  = mmg_params_vector(38) 
        C6  = mmg_params_vector(39) 
        C7  = mmg_params_vector(40)
        C10 = mmg_params_vector(41)
        ! Rudder
        tR             = mmg_params_vector(42)
        aH_rudder      = mmg_params_vector(43)
        xh_rudder_nd   = mmg_params_vector(44)
        kx_rudder      = mmg_params_vector(45)
        epsilon_rudder = mmg_params_vector(46)
        lR_nd          = mmg_params_vector(47)
        gammaN         = mmg_params_vector(48)
        gammaP         = mmg_params_vector(49)
        kx_rudder_reverse = mmg_params_vector(50)
        cpr_rudder     = mmg_params_vector(51)
        ! side thrusters    
        KT_bow_forward = mmg_params_vector(52)
        KT_bow_reverse = mmg_params_vector(53)
        aY_bow         = mmg_params_vector(54)
        aN_bow         = mmg_params_vector(55)

        KT_stern_forward = mmg_params_vector(56)
        KT_stern_reverse = mmg_params_vector(57)
        aY_stern         = mmg_params_vector(58)
        aN_stern         = mmg_params_vector(59)

        ! wind force coefficients
        XX0 = mmg_params_vector(60)
        XX1 = mmg_params_vector(61)
        XX3 = mmg_params_vector(62)
        XX5 = mmg_params_vector(63)
        YY1 = mmg_params_vector(64)
        YY3 = mmg_params_vector(65)
        YY5 = mmg_params_vector(66)
        NN1 = mmg_params_vector(67)
        NN2 = mmg_params_vector(68)
        NN3 = mmg_params_vector(69)
        KK1 = 0.0d0
        KK2 = 0.0d0
        KK3 = 0.0d0
        KK5 = 0.0d0
        !!!!!!!!!!!!!
        !!! main !!!!
        !!!!!!!!!!!!!

        ! Forward velocity   
        U_ship = sqrt(u_velo ** 2.0d0 + vm_velo ** 2.0d0)
        beta_hat = atan2(vm_velo, u_velo) 
        
        ! variables to add dimenstion
        D1_ad2 = 0.5d0 * rho_fresh * lpp ** 2.0d0 * draft
        D2_ad2 = 0.5d0 * rho_fresh * lpp ** 4.0d0 * draft
        Mass  = Mass_nd * D1_ad2
        MassX = MassX_nd * D1_ad2
        MassY = MassY_nd * D1_ad2
        IJzz  = IzzJzz_nd * D2_ad2
        xG    = xG_nd * lpp
        !
        if(abs(U_ship)< 1.0d-5)then
            v_nd = 0.0000001d0
            r_nd = 0.0000001d0
            U_ship   = 0.0000001d0
        else
            v_nd = vm_velo / U_ship  ! v -->> vm_velo (definition change 2018/4/5)
           r_nd = r_angvelo * lpp / U_ship
        end if
        
        !!!!!!!!!!!!!!!!!!!!!
        !!! Force of Hull
        !!!!!!!!!!!!!!!!!!!!!
        
        !integration parameters
        i_max = 200
        Y_ad = 0.0d0
        N_ad = 0.0d0
        
        do i = 1, i_max
            x0 = -0.5d0 + dble(i - 1) / dble(i_max)
            x1 = -0.5d0 + dble(i)  /dble(i_max)
            comp0 =vm_velo + C_rY * r_angvelo * lpp * x0
            comp1 =vm_velo + C_rY * r_angvelo * lpp * x1
            comp2 =vm_velo + C_rN * r_angvelo * lpp * x0
            comp3 =vm_velo + C_rN * r_angvelo * lpp * x1
            Y_ad = Y_ad + 0.5d0 * (abs(comp0) * comp0 + abs(comp1) * comp1) / dble(i_max)
            N_ad = N_ad + 0.5d0 * (abs(comp2) * comp2 * x0 + abs(comp3) * comp3 * x1) / dble(i_max)
        end do
        YHN_nd = - CD * Y_ad
        NHN_nd = - CD * N_ad
        
        XH = 0.5d0 * rho_fresh * lpp      * draft * &
                & ((X_0F_nd + (X_0A_nd - X_0F_nd) * (abs(beta_hat) / pi)) * u_velo * U_ship + Xvr_nd * vm_velo * r_angvelo * lpp)
        YH = 0.5d0 * rho_fresh * lpp      * draft *(Yv_nd * vm_velo * abs(u_velo) + Yr_nd * r_angvelo * lpp * u_velo + YHN_nd)
        NH = 0.5d0 * rho_fresh * lpp ** 2 * draft * (Nv_nd * vm_velo * u_velo + Nr_nd * r_angvelo * lpp * abs(u_velo) + NHN_nd)
    
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        !!! Force of Propeller !!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        Jsyn  = -0.35d0
        Jsyn0 = -0.06d0
        pitch = Dp * pitchr
    
        wP     = wP0 - tau_prop * abs(v_nd + xP_nd * r_nd) - CP_nd * (v_nd + xP_nd * r_nd) ** 2.0d0
        
        if(u_velo>0.0d0) then
            one_minus_wprop = 1.0d0 - wP
        else
            one_minus_wprop = 1.0d0 !u = up at 2nd and 4th quad.
        endif
        
        if( one_minus_wprop > 1.0d0 ) then
            one_minus_wprop = 1.0d0
        else if( one_minus_wprop <= 0.0d0 ) then
            one_minus_wprop = epsilon(1.0d0)
        end if
        uprop = u_velo * one_minus_wprop
        
        if(abs(n_prop) < epsilon(1.0d0))then
            Js = 1.0d+4
        elseif(abs(u_velo) < epsilon(1.0d0))then
            Js = epsilon(1.0d0) 
        else
            Js = u_velo / (Dp * n_prop)
        end if
    
        if(n_prop>=0.0d0 .and. u_velo>=0.0d0) then  ! 1st quad
            J_prop  = Js * one_minus_wprop
            KT = kt_coeff0 + kt_coeff1 * J_prop + kt_coeff2 * J_prop ** 2.0d0
            XP = rho_fresh * Dp ** 4.0d0 * n_prop ** 2.0d0 * (1.0d0 - t_prop) * KT
            YP = 0.0d0
            NP = 0.0d0
        elseif(n_prop>=0.0d0 .and. u_velo<0.0d0) then ! 2nd quad
            J_prop = Js
            KT = kt_coeff0 + kt_coeff1 * J_prop + kt_coeff2 * J_prop ** 2.0d0
            XP = rho_fresh * Dp ** 4.0d0 * n_prop ** 2.0d0 * (1.0d0 - t_prop) * KT
            YP = 0.5d0 * rho_fresh * lpp    * draft * (n_prop * pitch)**2.0d0 * (A6 * Js**2.0d0 + A7 * Js + A8)
            NP = 0.5d0 * rho_fresh * lpp**2.0d0 * draft * (n_prop * pitch)**2.0d0 * (B6 * Js**2.0d0 + B7 * Js + B8)
        else ! 3rd & 4th quad(n <0)
            if(u_velo>=0.0d0)then
                J_prop  = Js * one_minus_wprop
            else
                J_prop  = Js 
            endif
            
            if(Js >= C10)then
              XP = rho_fresh * n_prop ** 2.0d0 * Dp ** 4.0d0 * (C6 + C7 * Js)
            else
              XP = rho_fresh * n_prop ** 2.0d0 * Dp ** 4.0d0 * C3
            end if
            !
            if(Jsyn<=Js .and. Js<=Jsyn0)then
              YP = 0.5d0 * rho_fresh * lpp          * draft * (n_prop * Dp) ** 2.0d0 * (A1 + A2 * Js)
              NP = 0.5d0 * rho_fresh * lpp ** 2.0d0 * draft * (n_prop * Dp) ** 2.0d0 * (B1 + B2 * Js)
            elseif(Js<Jsyn)then
              YP = 0.5d0 * rho_fresh * lpp          * draft * (n_prop * Dp) ** 2.0d0 * (A3 + A4 * Js)
              NP = 0.5d0 * rho_fresh * lpp ** 2.0d0 * draft * (n_prop * Dp) ** 2.0d0 * (B3 + B4 * Js)
            elseif(Jsyn0<Js)then
              YP = 0.5d0 * rho_fresh * lpp          * draft * (n_prop * Dp) ** 2.0d0 * A5
              NP = 0.5d0 * rho_fresh * lpp ** 2.0d0 * draft * (n_prop * Dp) ** 2.0d0 * B5
            end if
        end if
        
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        !!! Force of Rudder    !!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        
        !  parameters 
        hight_rudder = sqrt(AREA * lambda_rudder)
        eta_rudder = Dp/hight_rudder
    
        xh_rudder = xh_rudder_nd * lpp
        lR  = lR_nd * lpp
        !    Fujii's formula
        fa = 6.13d0 * lambda_rudder / (2.25d0 + lambda_rudder)
        
        ! Calculate effective velocity to rudder uR
        kappa_rudder = kx_rudder / epsilon_rudder      
        one_minus_wrudder = epsilon_rudder * one_minus_wprop
        
        ! compute KT from XP to consider about 2nd,3rd,4th Quad
        if(n_prop>epsilon(1.0d0))then
            KT = XP / (rho_fresh * Dp**4.0d0 * n_prop**2.0d0 * (1-t_prop))
        elseif(abs(n_prop)>epsilon(1.0d0))then
            KT = XP / (rho_fresh * Dp**4.0d0 * n_prop**2.0d0 )
        else
            KT =0.0d0
        endif
        
        if( n_prop >= 0.0d0 .and. KT>0.0d0)then
           !uR = uprop * epsilon_rudder * sqrt(eta_rudder * (1.0d0 + kappa_rudder * &
           !    (sqrt(1.0d0 + 8.0d0 * KT / (pi * J_prop**2.0d0)) - 1.0d0))**2.0d0 + (1.0d0- eta_rudder))         !!! normal MMG model
            uR = epsilon_rudder * sqrt(eta_rudder * (uprop+ kappa_rudder *  &
                (sqrt(uprop**2 + 8.0d0 * KT * n_prop**2 * Dp **2/ (pi )) - uprop))**2 + (1- eta_rudder) * uprop**2) !!!<= normarl mmg model for low speed (Yoshimura's)
        else ! n<0 
        ! Kitagawa's model for uR in n<0
            if(u_velo<0.0d0)then  !4th quad
                uR = u_velo
            else                   !3rd quad
                urpr1 = u_velo * one_minus_wrudder + n_prop * Dp * kx_rudder_reverse * sqrt(8.0d0 * abs(KT)/ pi)
                urpr2 = u_velo * one_minus_wrudder
                ursq  = eta_rudder * sign(1.0d0, urpr1) * urpr1**2 + (1- eta_rudder) * urpr2**2.0d0 + cpr_rudder * u_velo**2.0d0
                uR =  sign(sqrt(abs(ursq)), ursq)    
            endif
        endif
        if(vm_velo+x_pos_rudder*r_angvelo>=0.0d0)then 
            vR = -1.0d0 * gammaP * (vm_velo + lR * r_angvelo)
        else
            vR = -1.0d0 *gammaN * (vm_velo + lR * r_angvelo)
        endif
        UUR = sqrt(uR ** 2.0d0 + vR ** 2.0d0)
        aR = delta_rudder - atan2(vR, uR)
        
        FN = 0.5d0 * rho_fresh * AREA * fa * UUR ** 2.0d0 * sin(aR)    
            !    Rudder forces and moments
        XR = - (1.0d0 - tR) * FN * sin(delta_rudder)
        YR = - (1.0d0 + aH_rudder) * FN * cos(delta_rudder)
        NR = - (x_pos_rudder + aH_rudder * xh_rudder) * FN * cos(delta_rudder)
        
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        !!! Force of Bow and stern thruster    !!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        Fr_number = abs(u_velo/sqrt(grav*lpp))
        ! Bow thruster
        if (n_bt >= 0.0d0)then
            YBT = rho_fresh * D_bthrust**4.0d0 * n_bt**2.0d0 * KT_bow_forward * (1.0d0 + aY_bow * Fr_number)
            NBT = rho_fresh * D_bthrust**4.0d0 * n_bt**2.0d0 * KT_bow_forward * (1.0d0 + aN_bow * Fr_number) * x_bowthrust_loc
        elseif (n_bt < 0.0d0)then       
            YBT = rho_fresh * D_bthrust**4.0d0 * n_bt**2.0d0 * KT_bow_reverse * (1.0d0 + aY_bow * Fr_number)
            NBT = rho_fresh * D_bthrust**4.0d0 * n_bt**2.0d0 * KT_bow_reverse * (1.0d0 + aN_bow * Fr_number) * x_bowthrust_loc
        end if     
        ! Stern thruster
        if (n_st >= 0.0d0)then
            YST = rho_fresh * D_sthrust**4.0d0 * n_st**2.0d0 * KT_stern_forward * (1.0d0 + aY_stern * Fr_number)
            NST = rho_fresh * D_sthrust**4.0d0 * n_st**2.0d0 * KT_stern_forward * (1.0d0 + aN_stern * Fr_number) * x_stnthrust_loc
        elseif (n_st<0.0d0)then        
            YST = rho_fresh * D_sthrust**4.0d0 * n_st**2.0d0 * KT_stern_reverse * (1.0d0 + aY_stern * Fr_number)
            NST = rho_fresh * D_sthrust**4.0d0 * n_st**2.0d0 * KT_stern_reverse * (1.0d0 + aN_stern * Fr_number) * x_stnthrust_loc
        end if     
        !  Restriction of force and moment due to forward speed
        if (abs( u_velo ) > Thruster_speed_max)then
            YBT = 0.0d0
            NBT = 0.0d0
            YST = 0.0d0
            NST = 0.0d0
        endif     


        !!!!!!!!!!!!!!!!!!!!!!!!!!
        !!! Force of wind      !!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        !   Save as Windcoef
        Windcoef(1) = AT;   Windcoef(2) = AL;  Windcoef(3) = AOD; Windcoef(4) = LCW;
        Windcoef(5) = LCBR; Windcoef(6) = HBR; Windcoef(7) = HC;  Windcoef(8) = SBW;
        !

        call Relative_Speed_and_Direction_Wind_or_Current(Wind_Direction, Wind_Velocity, psi, U_ship, beta_hat,&
        &                                                 Relative_Wind_Direction, Relative_Wind_Velocity)
        !   
        Angle_of_attack = 2.0d0 * pi - Relative_Wind_Direction
        
        !  Calculation of wind force and moment by Fujiwara's regresison
        ! call WindForce(Relative_Wind_Velocity, Angle_of_attack, &
        ! &              CXwind, CYwind, CNwind, CKwind,&
        ! &              FXwind, FYwind, FNwind, FKwind,&
        ! &              lpp, breadth, Lz, Windcoef)
        call WindForceOnly(Relative_Wind_Velocity, Angle_of_attack, CXwind, CYwind, &
        &                 CNwind, CKwind,&
        &                 FXwind, FYwind, FNwind, FKwind,&
        &                 lpp, Lz, Windcoef, XX0, XX1, XX3, XX5,&
        &                 YY1, YY3, YY5, NN1, NN2, NN3, KK1, KK2, KK3,KK5);
    
        XA = FXwind 
        YA = FYwind
        NA = FNwind
        !
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! Sum of all forces and moments
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        X = XH + XP + XR + XA
        Y = YH + YP + YR + YA + YBT + YST
        N = NH + NP + NR + NA + NBT + NST
    
        TempA1 = Mass + MassY
        TempA2 = xG * Mass
        TempA3 = Y - (Mass + MassX) * u_velo * r_angvelo
        TempB1 = IJzz + xG ** 2 * Mass
        TempB2 = xG * Mass
        TempB3 = N - xG * Mass * u_velo * r_angvelo
        u_dot  = (X + (Mass + MassY) * vm_velo * r_angvelo + xG * Mass * r_angvelo **2.0d0) / (Mass + MassX)
        vm_dot = (TempA3 * TempB1 - TempA2 * TempB3) / (TempA1 * TempB1 - TempA2 * TempB2)
        r_dot  = (TempA3 * TempA2 - TempB3 * TempA1) / (TempA2 * TempB2 - TempA1 * TempB1)
        !
        X_dot     = u_velo * cos(psi) - vm_velo * sin(psi)
        Y_dot     = u_velo * sin(psi) + vm_velo * cos(psi)
        psi_dot   = r_angvelo
        !
        time_derivative(1) = X_dot
        time_derivative(2) = u_dot
        time_derivative(3) = Y_dot
        time_derivative(4) = vm_dot
        time_derivative(5) = psi_dot
        time_derivative(6) = r_dot
        !
    end subroutine MMG_LowSpeed_model
    
    subroutine Relative_Speed_and_Direction_Wind_or_Current(zeta, UT, psi, U_ship,beta_hat, &
        Relative_Direction, Relative_Velocity)
        !
        !   Calculation of relative direction and speed
        !   zeta (Input:rad) : Direction of TRUE Wind or Current, earth fix coordinate(!! rad !!)
        !   UT (Input :m/s) : Speed of Wind or Current
        !   psi  (Input:deg) : Direction of Ship
        !   U_ship   (Input:m/s) : Speed of Ship
        !  beta_hat(input: rad): drift angle of ship (!! rad !!) (-pi, pi)
        !   Relative_Direction (Output: rad): Relative direction of Wind or Current (!! rad !!) 
        !   Relative_Velocity  (Output: m/s): Relative speed of Wind or Current
            !
            implicit none
            double precision, intent(in) :: zeta, UT, psi, U_ship,beta_hat
            double precision, intent(out) :: Relative_Direction,Relative_Velocity
            double precision :: chi,chi_beta, gamma_beta, RV
            !
            !  compute chi_beta(angle between ship traveling direction and true wind)
            chi = mod(3.0d0*pi - zeta + psi, 2.0d0*pi)
            chi_beta = chi+beta_hat
            if(chi_beta <0.0d0)then
                chi_beta = chi_beta + 2.0d0*pi 
            elseif(chi_beta >2.0d0*pi) then
                chi_beta = chi_beta - 2.0d0*pi        
            endif

            !  Calculation of relative direction and speed 
            Relative_Velocity = sqrt(UT * UT + U_ship * U_ship - 2.0D0 * UT * U_ship * cos(chi_beta))
            RV = (U_ship * U_ship + Relative_Velocity * Relative_Velocity - UT * UT) / 2.0D0 / U_ship / Relative_Velocity 

            if(abs(RV) > 1.0d0)then
                RV = dsign(1.0d0, RV) !   To avoid the value larger than 1.0 due to numerical error. 
            endif
            ! compute gamma_beta(angle between ship traveling direction and apparent wind)
            ! use if sentence because acos range is (0 pi) and gamma_beta is (0 2pi)
            if(chi_beta <= pi)then
                gamma_beta = acos(RV) 
            else
                gamma_beta = 2.0d0 * pi - acos(RV)
            endif

            ! compute relative direction(gamma_a) from gamma_beta
            Relative_Direction = gamma_beta + beta_hat


            if(abs(U_ship) < 1.0d-5)then
                Relative_Direction = mod(3.0d0*pi - chi, 2.0d0*pi)
            elseif(abs(UT) < 1.0d-5)then
                Relative_Direction = beta_hat
            endif
            
            if(Relative_Direction > 2.0d0*pi) then
                Relative_Direction = Relative_Direction - 2.0d0*pi
            elseif(Relative_Direction <0.0d0) then
                Relative_Direction = Relative_Direction + 2.0d0*pi 
            endif
    
    end subroutine Relative_Speed_and_Direction_Wind_or_Current

    subroutine WindForce(V, psi, CXwind, CYwind, CNwind, CKwind,&
        &                FXwind, FYwind, FNwind, FKwind,&
        &                lpp, breadth, Lz, Windcoef)
        !
        !   Estimation of Wind Forces and Moments acting on Ships by Fujiwara's method
        !  Journal of the Society of Naval Architects of Japan, Vol.183, pp.77-90, 1998
        !
        !       <<<<<       Potsitive definition of X wind force is backward       >>>>>
        !       <<<<<  Potsitive definition of psi is oppsite from usual MNVR case >>>>>
        !  
        !  ---------------------------------  Input  ---------------------------------
        !
        !  V ---- Wind velocity (m/s)
        !  psi -- Angle of attack (Potsitive definition of psi is oppsite from usual MNVR case) (rad)
        !  AT --- Transverse Projected area (m^2)
        !  AL --- Lateral Projected area (m^2)
        !  AOD -- Lateral Projected area of superstructure Ass and LNG tanks,
        !         container etc. on the deck (m^2)
        !         Here, Ass is defined as lateral Projected area of superstructure (m^2)
        !  C ---- Distance from midship section to center of lateral projected area (m)
        !  CBR -- Distance from midship section to center of the Ass (m)
        !  HBR -- Height from free surface to top of the superstructure (bridge) (m) 
        !  HC --- Height to center of lateral projected area (m)
        !
        !  -------------  Output (q = 0.5 * rhoa * U_ship ** 2, HL = AL / L)  -------------
        !
        !  CXwind --- CXwind = FXwind / (q * AT)
        !  CYwind --- CYwind = FYwind / (q * AL)
        !  CNwind --- CNwind = FNwind / (q * lpp * AT)
        !  CKwind --- CKwind = FKwind / (q * AL * HL)
        !  FXwind --- X directional wind force  acting on the ship hull
        !  FYwind --- Y directional wind force  acting on the ship hull
        !  FNwind --- N directional wind force  acting on the ship hull
        !  FKwind --- K directional wind moment acting on the ship hull  
        implicit none
        integer :: i
        double precision, intent(in) :: V, psi
        double precision, intent(in) :: lpp, breadth, Lz
        double precision, intent(out) :: CXwind, CYwind, CNwind, CKwind, FXwind, FYwind, FNwind, FKwind
        double precision :: rhoA, Gam
        double precision :: Windcoef(10)
        double precision :: AT, AL, AOD, LC, LCBR, HBR, HC, SBW
        double precision :: X00, X01, X02, X03
        double precision :: X10, X11, X12, X13, X14, X15, X16, X17
        double precision :: X30, X31, X32, X33, X34, X35, X36, X37
        double precision :: X50, X51, X52, X53
        double precision :: Y10, Y11, Y12, Y13, Y14, Y15
        double precision :: Y30, Y31, Y32, Y33, Y34, Y35, Y36
        double precision :: Y50, Y51, Y52, Y53, Y54, Y55, Y56
        double precision :: N10, N11, N12, N13, N14, N15, N16, N17, N18
        double precision :: N20, N21, N22, N23, N24, N25, N26, N27, N28
        double precision :: N30, N31, N32, N33
        double precision :: K10, K11, K12, K13, K14, K15, K16, K17, K18
        double precision :: K20, K21, K22, K23, K24, K25, K26, K27
        double precision :: K30, K31, K32, K33, K34, K35, K36
        double precision :: K50, K51, K52, K53, K54, K55
        double precision :: XX0, XX1, XX3, XX5
        double precision :: YY1, YY3, YY5
        double precision :: NN1, NN2, NN3
        double precision :: KK1, KK2, KK3,KK5
        !
        !	Difinition of physical values
        rhoA = 1.220d0 / grav
        Gam = 0.57721566490153286060d0
        !
        !   Wind coefficients
        AT   = Windcoef(1); AL  = Windcoef(2); AOD = Windcoef(3); LC  = Windcoef(4);
        LCBR = Windcoef(5); HBR = Windcoef(6); HC  = Windcoef(7); SBW = Windcoef(8);
        !   Coefficients of Fujiwara's regression
        !   X-directional Coefficients
        X00 = -0.330D0;  X01 =  0.293D0;  X02 =  0.0193D0; X03 =  0.6820D0;
        X10 = -1.353D0;  X11 =  1.700D0;  X12 =  2.8700D0; X13 = -0.4630D0;
        X14 = -0.570D0;  X15 = -6.640D0;  X16 = -0.0123D0; X17 =  0.0202D0;
        X30 =  0.830D0;  X31 = -0.413D0;  X32 = -0.0827D0; X33 = -0.5630D0;
        X34 =  0.804D0;  X35 = -5.670D0;  X36 =  0.0401D0; X37 = -0.1320D0;
        X50 =  0.0372D0; X51 = -0.0075D0; X52 = -0.1030D0; X53 =  0.0921D0;
        !   Y-directional Coefficients
        Y10 =  0.684d0;  Y11 =   0.717d0;  Y12 = -3.2200d0; Y13 =  0.0281d0; Y14 =  0.0661d0; Y15 =  0.2980d0;
        Y30 = -0.400d0;  Y31 =   0.282d0;  Y32 =  0.3070d0; Y33 =  0.0519d0; Y34 =  0.0526d0; Y35 = -0.0814d0; Y36 =  0.0582d0;
        Y50 =  0.122d0;  Y51 =  -0.166d0;  Y52 = -0.0054d0; Y53 = -0.0481d0; Y54 = -0.0136d0; Y55 =  0.0864d0; Y56 = -0.0297d0;
        !   N-directional Coefficients
        N10 =  0.2990d0; N11 =   1.710d0;  N12 =  0.183d0;  N13 = -1.09d0;   N14 = -0.0442d0; N15 = -0.289d0
        N16 =  4.24d0;  N17 = -0.0646d0; N18 =  0.0306d0;
        N20 =  0.1170d0; N21 =   0.123d0;  N22 = -0.323d0;  N23 =  0.0041d0; N24 = -0.166d0;  N25 = -0.0109d0;
        N26 =  0.174d0; N27 =  0.214d0;  N28 = -1.06d0
        N30 =  0.0230d0; N31 =   0.0385d0; N32 = -0.0339d0; N33 =  0.0023d0; 
        !   K-directional Coefficients
        K10 =  3.63d0;   K11 = -30.7d0;    K12 = 16.8d0;    K13 =  3.270d0;  K14 = -3.03d0;   K15 =  0.552d0;
        K16 = -3.03d0;   K17 = 1.82d0;   K18 = -0.224d0; 
        K20 = -0.480d0;  K21 =   0.166d0;  K22 =  0.318d0;  K23 =  0.132d0;  K24 = -0.148d0;
        K25 =  0.408d0;  K26 = -0.0394d0; K27 = 0.0041d0; 
        K30 =  0.164d0;  K31 =  -0.170d0;  K32 =  0.0803d0; K33 =  4.920d0;  K34 = -1.780d0;  K35 =  0.0404d0; K36 = -0.739d0; 
        K50 =  0.449d0;  K51 =  -0.148d0;  K52 = -0.0049d0; K53 = -0.396d0;  K54 = -0.0109d0; K55 = -0.0726d0;
        !  
        XX0 = X00 + X01 * (SBW * HBR / AT) + X02 * (LC / HC) + X03 * (AOD / lpp / lpp)
        XX1 = X10 + X11 * (AL / lpp / SBW) + X12 * (lpp * HC / AL) + X13 * (lpp * HBR / AL) + X14 * (AOD / AL) &
            + X15 * (AT / lpp / SBW) + X16 * (lpp * lpp / AT) + X17 * (lpp / HC)
        XX3 = X30 + X31 * (AL / lpp / HBR) + X32 * (AL / AT) + X33 * (lpp * HC / AL) + X34 * (AOD / AL) + X35 * (AOD / lpp / lpp) &
            + X36 * (LC / HC) + X37 * (LCBR / lpp)
        XX5 = X50 + X51 * (AL / AOD) + X52 * (LCBR / lpp) + X53 * (AL / lpp / SBW)
        !
        YY1 = Y10 + Y11 * (LCBR / lpp) + Y12 * (LC / lpp) + Y13 * (AL / AOD) + Y14 * (LC / HC) + Y15 * (AT / (SBW * HBR))
        YY3 = Y30 + Y31 * (AL / (lpp * SBW)) + Y32 * (lpp * HC / AL) + Y33 * (LCBR / lpp) + Y34 * (SBW / HBR) &
            & + Y35 * (AOD / AL) + Y36 * (AT / (SBW * HBR))
        YY5 = Y50 + Y51 * (AL / (lpp * SBW)) + Y52 * (lpp / HBR) + Y53 * (LCBR / lpp) + Y54 * (SBW ** 2 / AT) & 
            & + Y55 * (LC / lpp) + Y56 * (LC * HC / AL)
        !
        NN1 = N10 + N11 * (LC / lpp) + N12 * (lpp * HC / AL) + N13 * (AT / AL) + N14 * (LC / HC) &
            + N15 * (AL / (lpp * SBW)) + N16 * (AT / lpp ** 2) + N17 * (SBW ** 2 / AT) + N18 * (LCBR / lpp)
        NN2 = N20 + N21 * (LCBR / lpp) + N22 * (LC / lpp) + N23 * (AL / AOD) + N24 * (AT / SBW ** 2) &
            + N25 * (lpp / HBR) + N26 * (AT / (SBW * HBR)) + N27 * (AL / (lpp * SBW)) + N28 * (AL / lpp ** 2)
        NN3 = N30 + N31  *(LCBR / lpp) + N32 * (AT / (SBW * HBR)) + N33 * (AL / AT)
        !
        KK1 =K10 + K11 * (HBR / lpp) + K12 * (AT / (lpp * SBW)) + K13 * (lpp * HC / AL) + K14 * (LC / lpp) &
            + K15 * (LCBR / lpp) + K16 * (SBW / HBR) + K17 * (SBW ** 2 / AT) + K18 * (lpp / SBW)
        KK2 =K20 + K21 * (SBW / HBR) + K22 * (AT / SBW ** 2) + K23 * (AL / (lpp * HC)) + K24 * (LCBR / lpp) &
            + K25 * (HBR * LC / AL) + K26 * (lpp / SBW) + K27 * (lpp ** 2 / AL)
        KK3 =K30 + K31 * (SBW ** 2 / AT) + K32 * (LCBR / lpp) + K33 * (HC / lpp) + K34 * (AT / (lpp * SBW)) &
            + K35 * (lpp * SBW / AL) + K36 * (AOD / lpp ** 2)
        KK5 =K50 + K51 * (AL / (lpp * HC)) + K52 * (AL / AOD) + K53 * (AT / AL) + K54 * (lpp / SBW) + K55 * (AL / (lpp * SBW))
    

        !
        !  Cal of non-dimentionalized coefficients
        CXwind = XX0 + XX1 * cos(psi) + XX3 * cos(3.0D0 * psi) + XX5 * cos(5.0D0 * psi)
        CYwind = YY1 * sin(psi) + YY3 * sin(3.0D0 * psi) + YY5 * sin(5.0D0 * psi) ! YY5 is corrected (XX5 -->> YY5) 05/13/2019
        CNwind = NN1 * sin(psi) + NN2 * sin(2.0D0 * psi) + NN3 * sin(3.0D0 * psi)
        CKwind = KK1 * sin(psi) + KK2 * sin(2.0D0 * psi) + KK3 * sin(3.0D0 * psi) + KK5 * sin(5.0D0 * psi)
        !  Dimentionalization
        FXwind = CXwind * (0.5D0 * rhoA * V * V) * AT
        FYwind = CYwind * (0.5D0 * rhoA * V * V) * AL
        FNwind = CNwind * (0.5D0 * rhoA * V * V) * lpp *AL
        FKwind = CKwind * (0.5D0 * rhoA * V * V) * AL * (AL / lpp)
        !  Convert K morment around G
        FKwind = FKwind + FYwind * Lz
        CKwind = FKwind / ((0.5D0 * rhoA * V * V) * AL * (AL / lpp))
        !
        return
    end subroutine WindForce

    subroutine WindForceOnly(V, psi, CXwind, CYwind, CNwind, CKwind,&
        &                    FXwind, FYwind, FNwind, FKwind,&
        &                    lpp, Lz, Windcoef, XX0, XX1, XX3, XX5, YY1, YY3, YY5, NN1, NN2, NN3, KK1, KK2, KK3,KK5);
        
        !  CXwind --- CXwind = FXwind / (q * AT)
        !  CYwind --- CYwind = FYwind / (q * AL)
        !  CNwind --- CNwind = FNwind / (q * lpp * AT)
        !  CKwind --- CKwind = FKwind / (q * AL * HL)
        !  FXwind --- X directional wind force  acting on the ship hull
        !  FYwind --- Y directional wind force  acting on the ship hull
        !  FNwind --- N directional wind force  acting on the ship hull
        !  FKwind --- K directional wind moment acting on the ship hull  
        implicit none
        double precision, intent(in) :: V, psi;
        double precision, intent(in) :: lpp, Lz;
        double precision :: Windcoef(10);
        double precision :: AT, AL, AOD, LC, LCBR, HBR, HC, SBW;
        double precision, intent(out) :: CXwind, CYwind, CNwind, CKwind, FXwind, FYwind, FNwind, FKwind;
        double precision :: rhoA
        double precision, intent(in) :: XX0, XX1, XX3, XX5;
        double precision, intent(in) :: YY1, YY3, YY5;
        double precision, intent(in) :: NN1, NN2, NN3;
        double precision, intent(in) :: KK1, KK2, KK3,KK5;
        
        rhoA = 1.220d0 / grav;
        !
        !   Wind coefficients
        AT   = Windcoef(1); AL  = Windcoef(2); AOD = Windcoef(3); LC  = Windcoef(4);
        LCBR = Windcoef(5); HBR = Windcoef(6); HC  = Windcoef(7); SBW = Windcoef(8);
        !
        !  Cal of non-dimentionalized coefficients
        CXwind = XX0 + XX1 * cos(psi) + XX3 * cos(3.0D0 * psi) + XX5 * cos(5.0D0 * psi);
        CYwind = YY1 * sin(psi) + YY3 * sin(3.0D0 * psi) + YY5 * sin(5.0D0 * psi); ! YY5 is corrected (XX5 -->> YY5) 05/13/2019
        CNwind = NN1 * sin(psi) + NN2 * sin(2.0D0 * psi) + NN3 * sin(3.0D0 * psi);
        CKwind = KK1 * sin(psi) + KK2 * sin(2.0D0 * psi) + KK3 * sin(3.0D0 * psi) + KK5 * sin(5.0D0 * psi);
        !  Dimentionalization
        FXwind = CXwind * (0.5D0 * rhoA * V * V) * AT;
        FYwind = CYwind * (0.5D0 * rhoA * V * V) * AL;
        FNwind = CNwind * (0.5D0 * rhoA * V * V) * lpp *AL;
        FKwind = CKwind * (0.5D0 * rhoA * V * V) * AL * (AL / lpp);
        !  Convert K morment around G
        FKwind = FKwind + FYwind * Lz;
        CKwind = FKwind / ((0.5D0 * rhoA * V * V) * AL * (AL / lpp));
        !
        return
    end subroutine WindForceOnly
end module esso_osaka_vector_input