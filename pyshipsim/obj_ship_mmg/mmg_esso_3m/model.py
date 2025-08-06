import warnings
import numpy as np

from .py.model import MMGModel as pyMMGModel
from ..surfaceship import SurfaceShip
from ...utils import SpatialArray, TimeSpatialArray


try:
    from .f2py.model import MMGModel as f2pyMMGModel

    IMPORT_F2PY = True
except ImportError:
    print("ImportError: Import of esso_osaka_3m's f2py mpdule failed.")
    IMPORT_F2PY = False


class MMGEssoOsaka3m(SurfaceShip):

    def __init__(
        self,
        solve_method: str = "rk4",
        interp_method: str = "linear",
        polygon_type: str = "simple_ship",
        f2py: bool = True,
        **kwargs,
    ):
        super().__init__(
            solve_method=solve_method,
            interp_method=interp_method,
            polygon_type=polygon_type,
            **kwargs,
        )
        self.maneuver = f2pyMMGModel() if f2py and IMPORT_F2PY else pyMMGModel()
        if f2py and not IMPORT_F2PY:
            msg = "You can't use Esso_osaka_3m's f2py mpdule. Please compile fortran code."
            warnings.warn(msg)

    def regist_variables(self):
        super().regist_variables()
        args = {"obj_name": self.name, "var_type": "external_state"}
        args["var_name"] = f"{self.e_tag}delta_rudder [rad]"
        args["ub"], args["lb"] = np.deg2rad(35), -np.deg2rad(35)
        self.var_info.append({**args})
        args["var_name"] = f"{self.e_tag}n_prop [rps]"
        args["ub"], args["lb"] = 20, -20
        self.var_info.append({**args})
        args["var_name"] = f"{self.e_tag}true_wind_speed [m/s]"
        args["ub"], args["lb"] = np.inf, 0.0
        self.var_info.append({**args})
        args["var_name"] = f"{self.e_tag}true_wind_direction [rad]"
        args["ub"], args["lb"] = 2 * np.pi, 0.0
        self.var_info.append({**args})
        #
        self.tex_state_labels.update(
            {
                f"{self.e_tag}delta_rudder [rad]": "$\\delta \\ \\mathrm{(deg.)}$",
                f"{self.e_tag}n_prop [rps]": "$n_{\\mathrm{P}} \\ \\mathrm{(rps)}$",
                f"{self.e_tag}true_wind_speed [m/s]": "$U_{\\mathrm{T}} \\ \\mathrm{(m/s)}$",
                f"{self.e_tag}true_wind_direction [rad]": "$\\xi_{\\mathrm{T}} \\ \\mathrm{(deg.)}$",
            }
        )
        return self.var_info

    def time_invariant_ode_rhs(self, state, external_state):
        x, wT = state[0:6], external_state[2:4]
        u = np.array([*external_state[:2], 0, 0])
        return self.maneuver.ode_rhs(x, u, wT)

    def observe_state_seq(self, state_seq: TimeSpatialArray) -> TimeSpatialArray:
        if not hasattr(self, "mean") or not hasattr(self, "std"):
            mean = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            std = [0.03, 0.01, 0.03, 0.01, np.deg2rad(0.1), np.deg2rad(0.1)]
            self.mean, self.std = np.array(mean), np.array(std)
        noise = self.np_random.normal(
            loc=self.mean, scale=self.std, size=state_seq.shape
        )
        return state_seq + noise

    # def observe_state(self, state: SpatialArray) -> SpatialArray:
    #     if not hasattr(self, "mean") or not hasattr(self, "std"):
    #         mean = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #         std = [0.03, 0.01, 0.03, 0.01, np.deg2rad(0.1), np.deg2rad(0.1)]
    #         self.mean, self.std = np.array(mean), np.array(std)
    #     noise = self.np_random.normal(loc=self.mean, scale=self.std, size=state.shape)
    #     return state + noise


class MMGEssoOsaka3mWithThruster(MMGEssoOsaka3m):

    def regist_variables(self):
        super().regist_variables()
        args = {"obj_name": self.name, "var_type": "external_state"}
        args["var_name"] = f"{self.e_tag}n_bt [rps]"
        args["ub"], args["lb"] = 30, -30
        self.var_info.append({**args})
        args["var_name"] = f"{self.e_tag}n_st [rps]"
        args["ub"], args["lb"] = 30, -30
        self.var_info.append({**args})
        #
        self.tex_state_labels.update(
            {
                f"{self.e_tag}n_bt [rps]": "$n_{\\mathrm{BT}} \\ \\mathrm{(rps)}$",
                f"{self.e_tag}n_st [rps]": "$n_{\\mathrm{ST}} \\ \\mathrm{(rps)}$",
            }
        )
        return self.var_info

    def time_invariant_ode_rhs(self, state, external_state):
        x = state[0:6]
        u = np.array([*external_state[0:2], *external_state[4:6]])
        wT = external_state[2:4]
        x, u, wT = state[0:6], external_state[:4], external_state[4:6]
        return self.maneuver.ode_rhs(x, u, wT)
