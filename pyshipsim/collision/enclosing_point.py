import sys
import numpy as np

from ..utils.typing import PolyArray, BatchPointArray
from .base import CollisionCheckerBase


class EnclosingPointCollisionChecker(CollisionCheckerBase):
    """
    Collision checker that checks if any point of the ship is enclosed by an obstacle.
    """

    def collide(self, polygon: PolyArray, polygons: list[PolyArray]) -> bool:
        """
        Checks if any point of the ship polygon is enclosed by any obstacle polygon.

        Args:
            polygon (PolyArray): The ship polygon to check for collisions.
            polygons (list[PolyArray]): The list of obstacle polygons to check against.

        Returns:
            bool: True if any point of the ship polygon is enclosed by an obstacle polygon, False otherwise.
        """
        return enclosing_point_in_polygons(polygon, polygons)


def enclosing_point_in_polygons(polygon: PolyArray, polygons: list[PolyArray]) -> bool:
    """
    Check if a given polygon is enclosed within any polygon in a list of polygons.

    Args:
        polygon (PolyArray): The polygon to check.
        polygons (list[PolyArray]): A list of polygons to check against.

    Returns:
        bool: True if the polygon is enclosed within any of the polygons in the list, False otherwise.
    """
    return any(
        enclosing_point_in_polygon(polygon, other_polygon) for other_polygon in polygons
    )


def enclosing_point_in_polygon(points: BatchPointArray, polygon: PolyArray) -> bool:
    """
    Checks if any point in 'points' is enclosed by 'polygon'.

    Args:
        points (BatchPointArray): The points to check.
        polygon (PolyArray): The polygon to check against.

    Returns:
        bool: True if any point in 'points' is enclosed by 'polygon', False otherwise.
    """
    for point in points:
        l1 = polygon - point
        l2 = np.roll(l1, shift=-1, axis=0)
        cross_product = l1[:, 0] * l2[:, 1] - l1[:, 1] * l2[:, 0]
        dot_product = l1[:, 0] * l2[:, 0] + l1[:, 1] * l2[:, 1]
        total_angle = np.sum(np.arctan2(cross_product, dot_product))
        if np.abs(np.abs(total_angle) - 2 * np.pi) < sys.float_info.epsilon:
            return True
    return False
