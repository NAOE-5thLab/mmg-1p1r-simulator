from .harbor import Harbor


rectangle = [
    [100.0, 0.0],
    [-100.0, 0.0],
    [-100.0, 10.0],
    [100.0, 10.0],
]


class StraightBerth(Harbor):

    def __init__(self, scale=1, **kwargs):
        super().__init__(scale, **kwargs)
        self.add_xy_polygon(rectangle)


corner = [
    [0.0, 100.0],
    [0.0, 0.0],
    [-100.0, 0.0],
    [-100.0, -10.0],
    [10.0, -10.0],
    [10.0, 100.0],
]


class CornerBerth(Harbor):

    def __init__(self, scale=1, **kwargs):
        super().__init__(scale, **kwargs)
        self.add_xy_polygon(corner)
