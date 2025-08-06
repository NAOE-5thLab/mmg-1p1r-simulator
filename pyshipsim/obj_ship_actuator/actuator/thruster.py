import numpy as np
import matplotlib.pyplot as plt

from ...object import DynamicObject
from ...utils import TimeArray, SpatialArray, TimeSpatialArray


class BowThruster(DynamicObject):

    def __init__(
        self,
        solve_method: str = "rk4",
        interp_method: str = "previous",
        K: float = 20,
        T: float = 0.1,
        o_tag: str = "hat",
        e_tag: str = "cmd",
        **kwargs,
    ):
        super().__init__(
            solve_method=solve_method,
            interp_method=interp_method,
            o_tag=o_tag,
            e_tag=e_tag,
            **kwargs,
        )
        self.K, self.T = K, T

    def regist_variables(self):
        self.var_info = []
        args = {"obj_name": self.name, "ub": 30, "lb": -30}
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}n_bt [rps]", **args})
        args["var_type"] = "observation"
        self.var_info.append({"var_name": f"{self.o_tag}n_bt [rps]", **args})
        args["var_type"] = "external_state"
        self.var_info.append({"var_name": f"{self.e_tag}n_bt [rps]", **args})
        return self.var_info

    def time_invariant_ode_rhs(
        self, state: SpatialArray, external_state: SpatialArray
    ) -> SpatialArray:
        y, r = state, external_state
        dydt = np.clip((r - y) / self.T, -self.K, self.K)
        # dydt = self.K * np.sign(r - y)
        # dydt = (self.K * r - y) / self.T
        return dydt

    def observe_state_seq(self, state_seq: TimeSpatialArray) -> TimeSpatialArray:
        return state_seq

    # def observe_state(self, state: SpatialArray) -> SpatialArray:
    #     return state

    def subplot_timeseries(
        self,
        t_seq: TimeArray,
        state_seq: TimeSpatialArray,
        observation_seq: TimeSpatialArray,
        external_state_seq: TimeSpatialArray,
    ) -> tuple[plt.Figure, plt.Axes]:
        kwargs = {"sharex": True, "figsize": (4.8, 1.8), "tight_layout": True}
        fig, axes = plt.subplots(1, 1, **kwargs)
        kwargs = {"color": "black", "lw": 0.5, "label": "Simulation"}
        axes.plot(t_seq, state_seq, **kwargs)
        kwargs = {"color": "red", "lw": 0.5, "label": "Command"}
        axes.plot(t_seq, external_state_seq, **kwargs)
        # set y-axis limits
        axes.set_ylim(-31, 31)
        # set y-axis labels
        axes.set_ylabel("$n_{\\mathrm{BT}} \\ \\mathrm{(rps)}$")
        axes.set_xlabel("$t \\ \\mathrm{(s)}$")
        axes.legend()
        return fig, axes


class SternThruster(BowThruster):

    def regist_variables(self):
        self.var_info = []
        args = {"obj_name": self.name, "ub": 30, "lb": -30}
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}n_st [rps]", **args})
        args["var_type"] = "observation"
        self.var_info.append({"var_name": f"{self.o_tag}n_st [rps]", **args})
        args["var_type"] = "external_state"
        self.var_info.append({"var_name": f"{self.e_tag}n_st [rps]", **args})
        return self.var_info

    def subplot_timeseries(
        self,
        t_seq: TimeArray,
        state_seq: TimeSpatialArray,
        observation_seq: TimeSpatialArray,
        external_state_seq: TimeSpatialArray,
    ) -> tuple[plt.Figure, plt.Axes]:
        kwargs = {"sharex": True, "figsize": (4.8, 1.8), "tight_layout": True}
        fig, axes = plt.subplots(1, 1, **kwargs)
        kwargs = {"color": "black", "lw": 0.5, "label": "Simulation"}
        axes.plot(t_seq, state_seq, **kwargs)
        kwargs = {"color": "red", "lw": 0.5, "label": "Command"}
        axes.plot(t_seq, external_state_seq, **kwargs)
        # set y-axis limits
        axes.set_ylim(-31, 31)
        # set y-axis labels
        axes.set_ylabel("$n_{\\mathrm{ST}} \\ \\mathrm{(rps)}$")
        axes.set_xlabel("$t \\ \\mathrm{(s)}$")
        axes.legend()
        return fig, axes
