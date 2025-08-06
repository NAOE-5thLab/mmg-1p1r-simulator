import typing
import numpy as np
import numpy.typing as npt


__all__ = [
    "TimeArray",
    "SpatialArray",
    "TimeSpatialArray",
    "PolyArray",
    "BatchPointArray",
    "BatchSegmentArray",
    "BatchPolyArray",
    "PolyArray",
]

TimeArray = np.ndarray[(typing.Any), np.dtype[np.float64]]
SpatialArray = np.ndarray[(typing.Any), np.dtype[np.float64]]
TimeSpatialArray = np.ndarray[(typing.Any, typing.Any), np.dtype[np.float64]]

PolyArray = np.ndarray[(typing.Any, 2), np.dtype[np.float64]]
BatchPointArray = np.ndarray[(typing.Any, 2), np.dtype[np.float64]]
BatchSegmentArray = np.ndarray[(typing.Any, 2, 2), np.dtype[np.float64]]
BatchPolyArray = np.ndarray[(typing.Any, typing.Any, 2), np.dtype[np.float64]]

PolyArray = np.ndarray[(3), np.dtype[np.float64]]
