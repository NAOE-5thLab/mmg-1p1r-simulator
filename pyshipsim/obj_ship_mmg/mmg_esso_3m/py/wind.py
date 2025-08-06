# %%
import numpy as np


# from numba import jit
def irregularWind(wind_velo_ave, physical_time):
    """return instante velocity of irregular wind

    return instante velocity of irregular wind from average wind velocity using wind spectrum.
    process several timestep's data at once.(use np.ndarray).
    All Args and Returns are :x.shape = [number_of_timestep]
    Args:
        wind_velo_ave (float,ndarray): agerage velocity wind (m/s)

    Returns:
        wind_velo_instant (float, ndarray): instant velocity of irregular wind(m/s)
    """

    truncate_low_nd = 0.5
    truncate_high_nd = 10.0
    k_wind = 0.003
    div_omega = 200
    # take average to reduce component.
    peak_wind = np.average(np.pi * wind_velo_ave / (200.0 * np.sqrt(15.0)))
    truncate_low = truncate_low_nd * peak_wind
    truncate_high = truncate_high_nd * peak_wind
    d_omega_wind = (truncate_high - truncate_low) / div_omega
    omega_wind = np.empty(div_omega)
    spectrum_wind = np.empty(div_omega)
    amp_wind = np.empty(div_omega)

    for i in range(div_omega):
        omega_wind[i] = truncate_low + d_omega_wind * (i + 1)
        xd_wind = 600.0 * omega_wind[i] / (np.pi * np.average(wind_velo_ave))
        spectrum_wind[i] = 4.0 * k_wind * np.average(wind_velo_ave) ** 2 / omega_wind[
            i
        ] + xd_wind**2 / (1.0 + xd_wind**2) ** (4.0 / 3.0)
        amp_wind[i] = np.sqrt(2.0 * spectrum_wind[i] * d_omega_wind)

    wind_velo_instant = np.copy(wind_velo_ave)
    for j in range(np.size(physical_time)):
        for i in range(div_omega):
            wind_velo_instant[j] += amp_wind[i] * np.sin(
                omega_wind[i] * physical_time[j] + np.random.rand() * 2 * np.pi
            )

    return wind_velo_instant


# @jit('f8[:,:](f8[:,:],f8[:,:],f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8)',nopython=True)
def windForce(
    UA,
    AOA,
    AT,
    AL,
    Lz,
    rhoA,
    lpp,
    XX0,
    XX1,
    XX3,
    XX5,
    YY1,
    YY3,
    YY5,
    NN1,
    NN2,
    NN3,
    KK1,
    KK2,
    KK3,
    KK5,
):
    """compute force of wind affect to ship

    compute force of wind by apparent wind and ship geometry using Fujiwara's method (regression).
    process several timestep's data at once.(use np.ndarray)
    Args:
        UA (float, ndarray): Relative speed of Wind to ship(m/s), UA.shape = [number_of_timestep]
        AOA (float, ndarray): Angle of attack. Angle between ship heading and apparent wind direction.
                                conter-clock wise. [0, 2pi]. AOA.shape = [number_of_timestep]
        AT (float): Transverse Projected area (m^2)
        AL (float): Lateral Projected area (m^2)
        Lz(float): Acting position of sway force from center of gravity (it is necessary to calculate roll moment)
        rhoA(float): density of air (kgf/m3)
        lpp (float): length of ship (m)
        XX0,,,(float): Coefficient for Fujiwara's regression. given by function "windCoeff"
    Retruns:
        FX_wind (float, ndarray):
        FY_wind (float, ndarray):
        FN_wind (float, ndarray):
    """
    CXwind = np.empty((UA.shape[0], 1), dtype=np.float64)
    CYwind = np.empty((UA.shape[0], 1), dtype=np.float64)
    CNwind = np.empty((UA.shape[0], 1), dtype=np.float64)
    CKwind = np.empty((UA.shape[0], 1), dtype=np.float64)

    fwind = np.empty((UA.shape[0], 4), dtype=np.float64)
    # !  Cal of non-dimensionalized coefficients
    CXwind = XX0 + XX1 * np.cos(AOA) + XX3 * np.cos(3.0 * AOA) + XX5 * np.cos(5.0 * AOA)
    CYwind = YY1 * np.sin(AOA) + YY3 * np.sin(3.0 * AOA) + YY5 * np.sin(5.0 * AOA)
    CNwind = NN1 * np.sin(AOA) + NN2 * np.sin(2.0 * AOA) + NN3 * np.sin(3.0 * AOA)
    CKwind = (
        KK1 * np.sin(AOA)
        + KK2 * np.sin(2.0 * AOA)
        + KK3 * np.sin(3.0 * AOA)
        + KK5 * np.sin(5.0 * AOA)
    )

    # !  Dimentionalization
    FXwind = CXwind * (0.5 * rhoA * UA**2) * AT
    FYwind = CYwind * (0.5 * rhoA * UA**2) * AL
    FNwind = CNwind * (0.5 * rhoA * UA**2) * lpp * AL
    FKwind = CKwind * (0.5 * rhoA * UA**2) * AL * (AL / lpp)
    # !  Convert K morment around G
    FKwind = FKwind + FYwind * Lz
    CKwind = FKwind / ((0.5 * rhoA * UA**2) * AL * (AL / lpp))
    fwind = np.concatenate((FXwind, FYwind, FNwind, FKwind), axis=1)
    return fwind


def windCoeff(lpp, AT, AL, AOD, LCW, LCBR, HBR, HC, SBW, LZ):
    """coefficients of wind force

    Args:
        lpp (float): length of ship (m)
        AT (float): Transverse Projected area (m^2)
        AL (float): Lateral Projected area (m^2)
        AOD (float):Lateral Projected area of superstructure Ass and LNG tanks,
                    container etc. on the deck (m^2)
                    Here, Ass is defined as lateral Projected area of superstructure (m^2)
        LCW(float): Distance from midship section to center of lateral projected area (m)
        LCBR(float): Distance from midship section to center of the Ass (m)
        HBR(float): Height from free surface to top of the superstructure (bridge) (m)
        HC(float): Height to center of lateral projected area (m)
        SBW(float): SB for wind area (in usual, it coincides with SB, but it is defined for Trimaran vessel)
        Lz(float): Acting position of sway force from center of gravity (it is necessary to calculate roll moment)
    Retruns:
        return(float, ndarray): array of Coefficinet for Fujiwara's regression.
            [XX0, XX1, XX3, XX5, YY1, YY3, YY5, NN1, NN2, NN3, KK1, KK2, KK3, KK5]
    """
    # !   Coefficients of Fujiwara's regression
    # !   X-directional Coefficients
    X00 = -0.330
    X01 = 0.293
    X02 = 0.0193
    X03 = 0.6820
    X10 = -1.353
    X11 = 1.700
    X12 = 2.8700
    X13 = -0.4630
    X14 = -0.570
    X15 = -6.640
    X16 = -0.0123
    X17 = 0.0202
    X30 = 0.830
    X31 = -0.413
    X32 = -0.0827
    X33 = -0.5630
    X34 = 0.804
    X35 = -5.670
    X36 = 0.0401
    X37 = -0.1320
    X50 = 0.0372
    X51 = -0.0075
    X52 = -0.1030
    X53 = 0.0921
    #    Y-directional Coefficients
    Y10 = 0.684
    Y11 = 0.717
    Y12 = -3.2200
    Y13 = 0.0281
    Y14 = 0.0661
    Y15 = 0.2980
    Y30 = -0.400
    Y31 = 0.282
    Y32 = 0.3070
    Y33 = 0.0519
    Y34 = 0.0526
    Y35 = -0.0814
    Y36 = 0.0582
    Y50 = 0.122
    Y51 = -0.166
    Y52 = -0.0054
    Y53 = -0.0481
    Y54 = -0.0136
    Y55 = 0.0864
    Y56 = -0.0297
    # !   N-directional Coefficients
    N10 = 0.2990
    N11 = 1.710
    N12 = 0.183
    N13 = -1.09
    N14 = -0.0442
    N15 = -0.289
    N16 = 4.24
    N17 = -0.0646
    N18 = 0.0306
    N20 = 0.1170
    N21 = 0.123
    N22 = -0.323
    N23 = 0.0041
    N24 = -0.166
    N25 = -0.0109
    N26 = 0.174
    N27 = 0.214
    N28 = -1.06
    N30 = 0.0230
    N31 = 0.0385
    N32 = -0.0339
    N33 = 0.0023
    # !   K-directional Coefficients
    K10 = 3.63
    K11 = -30.7
    K12 = 16.8
    K13 = 3.270
    K14 = -3.03
    K15 = 0.552
    K16 = -3.03
    K17 = 1.82
    K18 = -0.224
    K20 = -0.480
    K21 = 0.166
    K22 = 0.318
    K23 = 0.132
    K24 = -0.148
    K25 = 0.408
    K26 = -0.0394
    K27 = 0.0041
    K30 = 0.164
    K31 = -0.170
    K32 = 0.0803
    K33 = 4.920
    K34 = -1.780
    K35 = 0.0404
    K36 = -0.739
    K50 = 0.449
    K51 = -0.148
    K52 = -0.0049
    K53 = -0.396
    K54 = -0.0109
    K55 = -0.0726

    ### compute coefficients for regression
    XX0 = X00 + X01 * (SBW * HBR / AT) + X02 * (LCW / HC) + X03 * (AOD / lpp / lpp)
    XX1 = (
        X10
        + X11 * (AL / lpp / SBW)
        + X12 * (lpp * HC / AL)
        + X13 * (lpp * HBR / AL)
        + X14 * (AOD / AL)
        + X15 * (AT / lpp / SBW)
        + X16 * (lpp * lpp / AT)
        + X17 * (lpp / HC)
    )
    XX3 = (
        X30
        + X31 * (AL / lpp / HBR)
        + X32 * (AL / AT)
        + X33 * (lpp * HC / AL)
        + X34 * (AOD / AL)
        + X35 * (AOD / lpp / lpp)
        + X36 * (LCW / HC)
        + X37 * (LCBR / lpp)
    )
    XX5 = X50 + X51 * (AL / AOD) + X52 * (LCBR / lpp) + X53 * (AL / lpp / SBW)

    YY1 = (
        Y10
        + Y11 * (LCBR / lpp)
        + Y12 * (LCW / lpp)
        + Y13 * (AL / AOD)
        + Y14 * (LCW / HC)
        + Y15 * (AT / (SBW * HBR))
    )
    YY3 = (
        Y30
        + Y31 * (AL / (lpp * SBW))
        + Y32 * (lpp * HC / AL)
        + Y33 * (LCBR / lpp)
        + Y34 * (SBW / HBR)
        + Y35 * (AOD / AL)
        + Y36 * (AT / (SBW * HBR))
    )
    YY5 = (
        Y50
        + Y51 * (AL / (lpp * SBW))
        + Y52 * (lpp / HBR)
        + Y53 * (LCBR / lpp)
        + Y54 * (SBW**2 / AT)
        + Y55 * (LCW / lpp)
        + Y56 * (LCW * HC / AL)
    )

    NN1 = (
        N10
        + N11 * (LCW / lpp)
        + N12 * (lpp * HC / AL)
        + N13 * (AT / AL)
        + N14 * (LCW / HC)
        + N15 * (AL / (lpp * SBW))
        + N16 * (AT / lpp**2)
        + N17 * (SBW**2 / AT)
        + N18 * (LCBR / lpp)
    )
    NN2 = (
        N20
        + N21 * (LCBR / lpp)
        + N22 * (LCW / lpp)
        + N23 * (AL / AOD)
        + N24 * (AT / SBW**2)
        + N25 * (lpp / HBR)
        + N26 * (AT / (SBW * HBR))
        + N27 * (AL / (lpp * SBW))
        + N28 * (AL / lpp**2)
    )
    NN3 = N30 + N31 * (LCBR / lpp) + N32 * (AT / (SBW * HBR)) + N33 * (AL / AT)

    KK1 = (
        K10
        + K11 * (HBR / lpp)
        + K12 * (AT / (lpp * SBW))
        + K13 * (lpp * HC / AL)
        + K14 * (LCW / lpp)
        + K15 * (LCBR / lpp)
        + K16 * (SBW / HBR)
        + K17 * (SBW**2 / AT)
        + K18 * (lpp / SBW)
    )
    KK2 = (
        K20
        + K21 * (SBW / HBR)
        + K22 * (AT / SBW**2)
        + K23 * (AL / (lpp * HC))
        + K24 * (LCBR / lpp)
        + K25 * (HBR * LCW / AL)
        + K26 * (lpp / SBW)
        + K27 * (lpp**2 / AL)
    )
    KK3 = (
        K30
        + K31 * (SBW**2 / AT)
        + K32 * (LCBR / lpp)
        + K33 * (HC / lpp)
        + K34 * (AT / (lpp * SBW))
        + K35 * (lpp * SBW / AL)
        + K36 * (AOD / lpp**2)
    )
    KK5 = (
        K50
        + K51 * (AL / (lpp * HC))
        + K52 * (AL / AOD)
        + K53 * (AT / AL)
        + K54 * (lpp / SBW)
        + K55 * (AL / (lpp * SBW))
    )

    return np.array(
        [XX0, XX1, XX3, XX5, YY1, YY3, YY5, NN1, NN2, NN3, KK1, KK2, KK3, KK5]
    )
