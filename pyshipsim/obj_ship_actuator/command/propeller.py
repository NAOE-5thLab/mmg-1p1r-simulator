from ...object import DynamicObject
from .base import ConstantCommander, UniformCommander, NormalCommander


class PropellerCommander(DynamicObject):

    def regist_variables(self):
        self.var_info = []
        args = {"obj_name": self.name, "ub": 20, "lb": -20}
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}n_prop [rps]", **args})
        return self.var_info


class PropellerConstantCommander(ConstantCommander, PropellerCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PropellerUniformCommander(UniformCommander, PropellerCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PropellerNormalCommander(NormalCommander, PropellerCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
