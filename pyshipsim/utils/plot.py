import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def cmapline(
    ax: plt.Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    palette: str = "viridis",
    **kwargs,
):
    cmap = plt.get_cmap(palette)
    colors = cmap(np.linspace(0, 1, len(y)))
    for i in range(len(x) - 1):
        ax.plot(x[i : i + 2], y[i : i + 2], color=colors[i], **kwargs)
    return ax
