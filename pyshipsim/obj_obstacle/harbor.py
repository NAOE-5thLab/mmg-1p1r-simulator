import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from .utils import latlon2local
from ..object import StaticObject
from ..utils import TimeSpatialArray, PolyArray


class Harbor(StaticObject):

    def __init__(
        self,
        scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.scale = scale
        self.polygons = []

    def add_xy_polygon(self, polygon: PolyArray | npt.ArrayLike) -> None:
        polygon = np.array(polygon)
        self.polygons.append(polygon * self.scale)

    def add_latlon_polygon(
        self, latlon_polygon: PolyArray | npt.ArrayLike, latlon_origin: npt.ArrayLike
    ) -> None:
        latlon_polygon = np.array(latlon_polygon)
        polygon = np.array([latlon2local(ll, latlon_origin) for ll in latlon_polygon])
        self.polygons.append(polygon * self.scale)

    def get_polygons(self) -> list[tuple[PolyArray, bool]]:
        return [[polygon, True] for polygon in self.polygons]

    def axes_x0y0(
        self,
        ax: plt.Axes,
        state_seq: TimeSpatialArray,
        observation_seq: TimeSpatialArray,
        scale: float = 1.0,
        plot_observation: bool = True,
    ) -> None:
        # polygons
        for polygon in self.polygons:
            polygon_over_L = polygon[:, [1, 0]] / scale
            kwargs = {"fill": True, "lw": 0.3, "fc": "#C8CCDE", "ec": "#0A0F60"}
            polygon = plt.Polygon(polygon_over_L, **kwargs)
            ax.add_patch(polygon)
