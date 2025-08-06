from ..object import SimulatedObject
from ..utils.typing import PolyArray
from .base import CollisionCheckerBase
from .enclosing_point import EnclosingPointCollisionChecker
from .segments_intersection import SegmentsIntersectCollisionChecker


class StrictCollisionChecker(CollisionCheckerBase):
    """
    Collision checker that combines EnclosingPointCollisionChecker and SegmentsIntersectCollisionChecker.
    """

    def __init__(self, *simobjs: SimulatedObject) -> None:
        super().__init__(*simobjs)
        self.checkers = [
            EnclosingPointCollisionChecker(*simobjs),
            SegmentsIntersectCollisionChecker(*simobjs),
        ]

    def collide(self, polygon: PolyArray, polygons: list[PolyArray]) -> bool:
        """
        Checks if the given polygon collides with any of the given polygons using both enclosing point and segment intersection methods.

        Args:
            polygon (PolyArray): The polygon to check for collisions.
            polygons (list[PolyArray]): The list of polygons to check against.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        for checker in self.checkers:
            if checker.collide(polygon, polygons):
                return True
        return False
