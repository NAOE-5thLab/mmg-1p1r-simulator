import numpy as np
from ...object import DynamicObject
from .base import ConstantCommander, UniformCommander, NormalCommander


class RudderCommander(DynamicObject):

    def regist_variables(self):
        self.var_info = []
        args = {"obj_name": self.name, "ub": np.deg2rad(35), "lb": -np.deg2rad(35)}
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}delta_rudder [rad]", **args})
        return self.var_info


class RudderConstantCommander(ConstantCommander, RudderCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RudderUniformCommander(UniformCommander, RudderCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RudderNormalCommander(NormalCommander, RudderCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VecTwinRudderCommander(DynamicObject):

    def regist_variables(self):
        self.var_info = []
        args = {"obj_name": self.name}
        args["var_type"] = "state"
        args["ub"], args["lb"] = np.deg2rad(35), -np.deg2rad(105)
        self.var_info.append({"var_name": f"{self.s_tag}delta_rudder_p [rad]", **args})
        args["ub"], args["lb"] = np.deg2rad(105), -np.deg2rad(35)
        self.var_info.append({"var_name": f"{self.s_tag}delta_rudder_s [rad]", **args})
        return self.var_info


class VecTwinRudderConstantCommander(ConstantCommander, VecTwinRudderCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VecTwinRudderUniformCommander(UniformCommander, VecTwinRudderCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VecTwinRudderNormalCommander(NormalCommander, VecTwinRudderCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
