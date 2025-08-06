
import numpy as np
from ...object import DynamicObject
from .base import ConstantCommander, UniformCommander, NormalCommander


class PitchCommander(DynamicObject):

    def regist_variables(self):
        self.var_info = []
        args = {"obj_name": self.name, "ub": np.deg2rad(20), "lb": -np.deg2rad(20)}
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}pitch_ang [rad]", **args})
        return self.var_info


class PitchConstantCommander(ConstantCommander, PitchCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PitchUniformCommander(UniformCommander, PitchCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PitchNormalCommander(NormalCommander, PitchCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
