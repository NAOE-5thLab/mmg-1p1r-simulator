import numpy as np
import numpy.typing as npt

from ..utils.typing import PolyArray, BatchSegmentArray, BatchPointArray
from .base import CollisionCheckerBase


class SegmentsIntersectCollisionChecker(CollisionCheckerBase):
    """
    Collision checker that checks if any segment of the ship intersects with any obstacle segment.
    """

    def collide(self, polygon: PolyArray, polygons: list[PolyArray]) -> bool:
        """
        Checks if any segment of the ship polygon intersects with any obstacle segment.

        Args:
            polygon (PolyArray): The ship polygon to check for collisions.
            polygons (list[PolyArray]): The list of obstacle polygons to check against.

        Returns:
            bool: True if any segment of the ship polygon intersects with any obstacle segment, False otherwise.
        """
        return segments_intersect_of_polygon_and_polygons(polygon, polygons)


def segments_intersect_of_polygon_and_polygons(
    polygon: PolyArray, polygons: PolyArray | list[PolyArray]
) -> bool:
    """
    Checks if any segment of the polygon intersects with any segment of the polygons.

    Args:
        polygon (PolyArray): The polygon to check for intersections.
        polygons (list[PolyArray]): The list of polygons to check against.

    Returns:
        bool: True if any segment of the polygon intersects with any segment of the polygons, False otherwise.
    """
    if len(polygons) == 0 or len(polygon) == 0:
        return False
    ship_segments = polygon2segments(polygon)
    if isinstance(polygons, np.ndarray):
        obstacle_segments = polygon2segments(polygons)
    else:
        obstacle_segments_list = [polygon2segments(p) for p in polygons]
        obstacle_segments = np.concatenate(obstacle_segments_list, axis=0)
    return segments_intersect(ship_segments, obstacle_segments)


def polygon2segments(polygon: PolyArray) -> BatchSegmentArray:
    """
    Converts a polygon to its segments.

    Args:
        polygon (PolyArray): The polygon to convert.

    Returns:
        BatchSegmentArray: The segments of the polygon.
    """
    shifted_polygon = np.roll(polygon, shift=-1, axis=0)
    segments = np.stack((polygon, shifted_polygon), axis=1)
    return segments


def segments_intersect(
    segments_A: BatchSegmentArray, segments_B: BatchSegmentArray
) -> bool:
    """
    Checks if any segment in 'segments_A' intersects with any segment in 'segments_B'.

    Args:
        segments_A (BatchSegmentArray): The first set of segments.
        segments_B (BatchSegmentArray): The second set of segments.

    Returns:
        bool: True if any segment in 'segments_A' intersects with any segment in 'segments_B', False otherwise.
    """
    PointsP1, PointsQ1, PointsP2, PointsQ2 = [], [], [], []
    for i in range(len(segments_A)):
        segments_B_num = len(segments_B)
        PointsP1.append(np.tile(segments_A[i, 0, :], (segments_B_num, 1)))
        PointsQ1.append(np.tile(segments_A[i, 1, :], (segments_B_num, 1)))
        PointsP2.append(segments_B[:, 0, :])
        PointsQ2.append(segments_B[:, 1, :])
    PointsP1 = np.concatenate(PointsP1, axis=0)
    PointsQ1 = np.concatenate(PointsQ1, axis=0)
    PointsP2 = np.concatenate(PointsP2, axis=0)
    PointsQ2 = np.concatenate(PointsQ2, axis=0)

    ori1 = orientation(PointsP1, PointsQ1, PointsP2)
    ori2 = orientation(PointsP1, PointsQ1, PointsQ2)
    ori3 = orientation(PointsP2, PointsQ2, PointsP1)
    ori4 = orientation(PointsP2, PointsQ2, PointsQ1)
    if np.any((ori1 != ori2) & (ori3 != ori4)):
        return True

    if np.any(ori1 == 0):
        on1 = onSegment(PointsP1, PointsP2, PointsQ1)
        if np.any((ori1 == 0) & (on1 == 1)):
            return True
    if np.any(ori2 == 0):
        on2 = onSegment(PointsP1, PointsQ2, PointsQ1)
        if np.any((ori2 == 0) & (on2 == 1)):
            return True
    if np.any(ori3 == 0):
        on3 = onSegment(PointsP2, PointsP1, PointsQ2)
        if np.any((ori3 == 0) & (on3 == 1)):
            return True
    if np.any(ori4 == 0):
        on4 = onSegment(PointsP2, PointsQ1, PointsP2)
        if np.any((ori4 == 0) & (on4 == 1)):
            return True
    return False


def orientation(
    A: BatchPointArray, B: BatchPointArray, C: BatchPointArray
) -> npt.NDArray[np.int64]:
    """
    Determines the orientation of the triplet (A, B, C).

    Args:
        A (BatchPointArray): The first point.
        B (BatchPointArray): The second point.
        C (BatchPointArray): The third point.

    Returns:
        npt.NDArray[np.int64]: An array indicating the orientation of each triplet.
    """
    fir = (B[:, 1] - A[:, 1]) * (C[:, 0] - B[:, 0])
    sec = (B[:, 0] - A[:, 0]) * (C[:, 1] - B[:, 1])
    cross = fir - sec
    ori = np.where(cross < 0, -1, 0)
    ori = np.where(cross > 0, 1, ori)
    return ori


def onSegment(
    A: BatchPointArray, B: BatchPointArray, C: BatchPointArray
) -> npt.NDArray[np.int64]:
    """
    Checks if point A lies on the segment BC.

    Args:
        A (BatchPointArray): The point to check.
        B (BatchPointArray): One endpoint of the segment.
        C (BatchPointArray): The other endpoint of the segment.

    Returns:
        npt.NDArray[np.int64]: An array indicating whether point A lies on the segment BC.
    """
    BC = np.array([B, C])
    BCmax = np.max(BC, axis=0)
    BCmin = np.min(BC, axis=0)
    cond1 = (A[:, 0] <= BCmax[:, 0]) & (A[:, 0] >= BCmin[:, 0])
    cond2 = (A[:, 1] <= BCmax[:, 1]) & (A[:, 1] >= BCmin[:, 1])
    return np.where(cond1 & cond2, 1, 0)
