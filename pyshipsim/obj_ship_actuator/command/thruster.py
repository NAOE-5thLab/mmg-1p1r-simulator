from ...object import DynamicObject
from .base import ConstantCommander, UniformCommander, NormalCommander


class BowThrusterCommander(DynamicObject):

    def regist_variables(self):
        self.var_info = []
        args = {"obj_name": self.name, "ub": 30, "lb": -30}
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}n_bt [rps]", **args})
        return self.var_info


class BowThrusterConstantCommander(ConstantCommander, BowThrusterCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BowThrusterUniformCommander(UniformCommander, BowThrusterCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BowThrusterNormalCommander(NormalCommander, BowThrusterCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SternThrusterCommander(DynamicObject):

    def regist_variables(self):
        self.var_info = []
        args = {"obj_name": self.name, "ub": 30, "lb": -30}
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}n_st [rps]", **args})
        return self.var_info


class SternThrusterConstantCommander(ConstantCommander, SternThrusterCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SternThrusterUniformCommander(UniformCommander, SternThrusterCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SternThrusterNormalCommander(NormalCommander, SternThrusterCommander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
