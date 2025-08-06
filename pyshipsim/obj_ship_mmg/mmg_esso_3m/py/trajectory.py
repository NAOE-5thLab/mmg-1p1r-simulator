import numpy as np
def ship_coo(x, L, B, scale = 2):
    """
    Args:
        x (float, ndarray): [x_position_mid, y_position_mid, psi]
        L,B (Float): Length and breadth of ship
        scale: 
    Returns:
        x1: F.P. of ship
        x2: fore starboard
        x3: aft starboard
        x4: aft port
        x5: fore port
    """
    xg, yg, phi = x
    #phi = phi * np.pi/180
    Lrate = 0.6
    x1 = np.array([xg + scale * L/2.0 * np.cos(phi), yg + scale * L/2.0 * np.sin(phi)])
    x2 = np.array([xg + scale * (L/2.0 * Lrate * np.cos(phi) - B/2.0 * np.sin(phi)), yg + scale * (L/2.0 * Lrate * np.sin(phi) + B/2.0 * np.cos(phi))])
    x3 = np.array([xg + scale * ((-1) * L/2.0 * np.cos(phi) - B/2.0 * np.sin(phi)), yg + scale * ((-1) * L/2.0 * np.sin(phi) + B/2.0 * np.cos(phi))])
    x4 = np.array([xg + scale * ((-1) * L/2.0 * np.cos(phi) + B/2.0 * np.sin(phi)), yg + scale * ((-1) * L/2.0 * np.sin(phi) - B/2.0 * np.cos(phi))])
    x5 = np.array([xg + scale * (L/2.0 * Lrate * np.cos(phi) + B/2.0 * np.sin(phi)), yg + scale * (L/2.0 * Lrate * np.sin(phi) - B/2.0 * np.cos(phi))])
    return x1, x2, x3, x4, x5