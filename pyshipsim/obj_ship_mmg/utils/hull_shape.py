import typing
import numpy as np
import numpy.typing as npt

from ...utils import SpatialArray, PolyArray

__all__ = [
    "simple_ship_poly",
    "detail_ship_poly",
    "rectangle_ship_poly",
    "ellipse_ship_poly",
    "ship_domain_miyauchi_poly",
]


simple_ship_poly_unit = np.array(
    [
        [-0.5, -0.5],
        [0.6 * 0.5, -0.5],
        [0.5, 0.0],
        [0.6 * 0.5, 0.5],
        [-0.5, 0.5],
    ]
)


def simple_ship_poly(x: SpatialArray, L: float, B: float) -> PolyArray:
    poly = simple_ship_poly_unit * np.array([L, B])
    return rotation2D(poly, x[4]) + x[[0, 2]]


detail_ship_poly_unit = np.array(
    [
        [-0.5000, -0.2966],  # [-0.5000000000000000, -0.2966136692132980],
        [-0.4808, -0.3266],  # [-0.4807692307692310, -0.3265825005162090],
        [-0.4487, -0.3723],  # [-0.4487179487179490, -0.3722829919471400],
        [-0.4167, -0.4150],  # [-0.4166666666666670, -0.4150238736320460],
        [-0.3846, -0.4460],  # [-0.3846153846153850, -0.4459710014453850],
        [-0.3526, -0.4721],  # [-0.3525641025641030, -0.4720615589510630],
        [-0.3205, -0.4889],  # [-0.3205128205128200, -0.4889192360107370],
        [-0.2885, -0.4980],  # [-0.2884615384615380, -0.4979829960768120],
        [-0.2564, -0.5000],  # [-0.2564102564102560, -0.5000000000000000],
        [-0.1923, -0.5000],  # [-0.1923076923076920, -0.5000000000000000],
        [-0.1282, -0.5000],  # [-0.1282051282051280, -0.5000000000000000],
        [-0.0641, -0.5000],  # [-0.0641025641025641, -0.5000000000000000],
        [0.0000, -0.5000],  # [0.0000000000000000, -0.5000000000000000],
        [0.0619, -0.5000],  # [0.0618641390358615, -0.5000000000000000],
        [0.1237, -0.5000],  # [0.1237282780717230, -0.5000000000000000],
        [0.1856, -0.5000],  # [0.1855924171075840, -0.5000000000000000],
        [0.2475, -0.5000],  # [0.2474565561434460, -0.5000000000000000],
        [0.2784, -0.5000],  # [0.2783886256613770, -0.5000000000000000],
        [0.3093, -0.4983],  # [0.3093206951793070, -0.4983129651042740],
        [0.3403, -0.4825],  # [0.3402527646972380, -0.4825415135246750],
        [0.3712, -0.4521],  # [0.3711848342151690, -0.4521218521577530],
        [0.4021, -0.4120],  # [0.4021169037330990, -0.4119625211645670],
        [0.4330, -0.3510],  # [0.4330489732510300, -0.3509664732603760],
        [0.4640, -0.2376],  # [0.4639810427689610, -0.2375627359074950],
        [0.4949, -0.0858],  # [0.4949131122868920, -0.0858179475531695],
        [0.5000, 0.0000],  # [0.5000000000000000, 0.0000000000000000],
        [0.4949, 0.0858],  # [0.4949131122868920, 0.0858179475531695],
        [0.4640, 0.2376],  # [0.4639810427689610, 0.2375627359074950],
        [0.4330, 0.3510],  # [0.4330489732510300, 0.3509664732603760],
        [0.4021, 0.4120],  # [0.4021169037330990, 0.4119625211645670],
        [0.3712, 0.4521],  # [0.3711848342151690, 0.4521218521577530],
        [0.3403, 0.4825],  # [0.3402527646972380, 0.4825415135246750],
        [0.3093, 0.4983],  # [0.3093206951793070, 0.4983129651042740],
        [0.2784, 0.5000],  # [0.2783886256613770, 0.5000000000000000],
        [0.2475, 0.5000],  # [0.2474565561434460, 0.5000000000000000],
        [0.1856, 0.5000],  # [0.1855924171075840, 0.5000000000000000],
        [0.1237, 0.5000],  # [0.1237282780717230, 0.5000000000000000],
        [0.0619, 0.5000],  # [0.0618641390358615, 0.5000000000000000],
        [0.0000, 0.5000],  # [0.0000000000000000, 0.5000000000000000],
        [-0.0641, 0.5000],  # [-0.0641025641025641, 0.5000000000000000],
        [-0.1282, 0.5000],  # [-0.1282051282051280, 0.5000000000000000],
        [-0.1923, 0.5000],  # [-0.1923076923076920, 0.5000000000000000],
        [-0.2564, 0.5000],  # [-0.2564102564102560, 0.5000000000000000],
        [-0.2885, 0.4980],  # [-0.2884615384615380, 0.4979829960768120],
        [-0.3205, 0.4889],  # [-0.3205128205128200, 0.4889192360107370],
        [-0.3526, 0.4721],  # [-0.3525641025641030, 0.4720615589510630],
        [-0.3846, 0.4460],  # [-0.3846153846153850, 0.4459710014453850],
        [-0.4167, 0.4150],  # [-0.4166666666666670, 0.4150238736320460],
        [-0.4487, 0.3723],  # [-0.4487179487179490, 0.3722829919471400],
        [-0.4808, 0.3266],  # [-0.4807692307692310, 0.3265825005162090],
        [-0.5000, 0.2966],  # [-0.5000000000000000, 0.2966136692132980],
        # [-0.5000, -0.2966],  # [-0.5000000000000000, -0.2966136692132980],
    ]
)


def detail_ship_poly(x: SpatialArray, L: float, B: float) -> PolyArray:
    poly = detail_ship_poly_unit * np.array([L, B])
    return rotation2D(poly, x[4]) + x[[0, 2]]


rectangle_ship_poly_unit = np.array(
    [
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
    ]
)


def rectangle_ship_poly(x: SpatialArray, L: float, B: float) -> PolyArray:
    poly = rectangle_ship_poly_unit * np.array([L, B])
    return rotation2D(poly, x[4]) + x[[0, 2]]


def ellipse_ship_poly(
    x: SpatialArray, L: float, B: float, split_num: int = 20
) -> PolyArray:
    """Ellipse polygon function

    Args:
        eta (np.ndarray[(3), np.dtype[np.float64]]): Position and headding angle (x_0, y_0, psi)
        Lpp (float): Ship length
        B (float): Ship breadth
        split_num (int, optional): Defaults to 10.

    Returns:
        np.ndarray[(typing.Any, 2), np.dtype[np.float64]]: Polygon array
    """
    alphas = np.arange(-np.pi, np.pi, 2 * np.pi / split_num)
    pos_x = 0.5 * L * np.cos(alphas)
    pos_y = 0.5 * B * np.sin(alphas)
    points = np.concatenate([pos_x[:, np.newaxis], pos_y[:, np.newaxis]], axis=1)
    return rotation2D(poly, x[4]) + x[[0, 2]]


def ship_domain_miyauchi_poly(
    x: SpatialArray,
    Lpp: float,
    B: float,
    W: float,
    scale: float,
    split_num: int = 10,
    split_type: int = 0,
) -> np.ndarray[(typing.Any, 2), np.dtype[np.float64]]:
    """This function returns the polygon that approximates the ship domain proposed by Miyauchi (2022)

    Args:
        eta (np.ndarray[(3), np.dtype[np.float64]]): Position and headding angle (x_0, y_0, psi)
        nu (np.ndarray[(3), np.dtype[np.float64]]): Velocity and angle velocity (u, v_m, r)
        Lpp (float): Ship length
        B (float): Ship breadth
        W (float):  Minimum passage width
        scale (float): Scale compared to real ship
        split_num (int, optional): Defaults to 10.
        split_type (int, optional): Defaults to 0.

    Returns:
        np.ndarray[(typing.Any, 2), np.dtype[np.float64]]: ship shape polygon

    References:
        Yoshiki Miyauchi et al.
        Optimization on planning of trajectory and control of autonomous berthing and unberthing for the realistic port geometry,
        Ocean Engineering, Volume 245, 2022, 110390, ISSN 0029-8018, https://doi.org/10.1016/j.oceaneng.2021.110390.
    """
    kt = (1852 / 3600) * np.sqrt(scale)
    U_min = 1 * kt
    U_max = 6 * kt
    U = np.linalg.norm(x[[1, 3]])
    U = np.max([U, U_min])
    U = np.min([U, U_max])
    #
    Lx_min = 0.75 * Lpp
    Ly_min = B
    Lx_max_fwd = 0.75 * W
    Lx_max_aft = 0.5 * W
    Ly_max = 0.25 * W
    Lx_fwd = (Lx_max_fwd - Lx_min) * (U - U_min) / (U_max - U_min) + Lx_min
    Lx_aft = (Lx_max_aft - Lx_min) * (U - U_min) / (U_max - U_min) + Lx_min
    Ly = (Ly_max - Ly_min) * (U - U_min) / (U_max - U_min) + Ly_min
    #
    alphas = np.arange(-np.pi, np.pi, 2 * np.pi / split_num)
    if split_type == 0:
        pos_x_fwd = Lx_fwd * np.cos(alphas)
        pos_x_aft = Lx_aft * np.cos(alphas)
        pos_x = np.where((0.0 <= np.cos(alphas)), pos_x_fwd, pos_x_aft)
        pos_y = Ly * np.sin(alphas)
    elif split_type == 1:
        pos_x_fwd = (Lx_fwd * Ly) / np.sqrt(Ly**2 + Lx_fwd**2 * np.tan(alphas) ** 2)
        pos_x_aft = (Lx_aft * Ly) / np.sqrt(Ly**2 + Lx_aft**2 * np.tan(alphas) ** 2)
        pos_x = np.where((0.0 <= np.cos(alphas)), pos_x_fwd, pos_x_aft)
        pos_y_fwd = (Lx_fwd * Ly * np.tan(alphas)) / np.sqrt(
            Ly**2 + Lx_fwd**2 * np.tan(alphas) ** 2
        )
        pos_y_aft = (Lx_aft * Ly * np.tan(alphas)) / np.sqrt(
            Ly**2 + Lx_aft**2 * np.tan(alphas) ** 2
        )
        pos_y = np.where((0.0 <= np.cos(alphas)), pos_y_fwd, pos_y_aft)
    else:
        print("Error : split_type is invalid")
    points = np.concatenate([pos_x[:, np.newaxis], pos_y[:, np.newaxis]], axis=1)
    return rotation2D(poly, x[4]) + x[[0, 2]]


def rotation2D(
    pos: npt.NDArray[np.float64],
    psi: float,
) -> npt.NDArray[np.float64]:
    A = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    return np.dot(pos, A.T)
