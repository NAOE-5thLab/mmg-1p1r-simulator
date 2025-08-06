from ..utils.typing import PolyArray, TimeArray
from ..object import SimulatedObject


class CollisionCheckerBase:
    """
    Base class for collision checkers.
    """

    def __init__(self, *simobjs: SimulatedObject) -> None:
        self.simobjs = simobjs

    def check(self, idx: int = None) -> bool:
        """
        Checks for collisions among the simulated objects at a specific time index.

        Args:
            idx (int, optional): The time index to check for collisions. Defaults to None.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        polygons_objs, allowable_flag_objs = [], []
        for simobj in self.simobjs:
            if hasattr(simobj, "get_polygons") and hasattr(simobj, "get_state"):
                state = simobj.get_state(idx=idx)
                for polygon, allowable_flag in simobj.get_polygons(state):
                    polygons_objs.append(polygon)
                    allowable_flag_objs.append(allowable_flag)
            elif hasattr(simobj, "get_polygons"):
                for polygon, allowable_flag in simobj.get_polygons():
                    polygons_objs.append(polygon)
                    allowable_flag_objs.append(allowable_flag)
        #
        for i, polygons in enumerate(polygons_objs):
            if not allowable_flag_objs[i]:
                other_polygons = polygons_objs[:i] + polygons_objs[i + 1 :]
                if self.collide(polygons, other_polygons):
                    return True
        return False

    def check_seq(self, t_seq: TimeArray) -> list[bool]:
        """
        Checks for collisions among the simulated objects over a sequence of time steps.

        Args:
            t_seq (TimeArray): The sequence of time steps.

        Returns:
            list[bool]: A list indicating whether a collision is detected at each time step.
        """
        return [self.check(ti) for ti in range(len(t_seq))]

    def collide(self, polygon: PolyArray, polygons: list[PolyArray]) -> bool:
        """
        Checks if a polygon collides with any other polygons.

        Args:
            polygon (PolyArray): The polygon to check for collisions.
            polygons (list[PolyArray]): The list of other polygons to check against.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
