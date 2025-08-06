import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from ...object import DynamicObject
from ...utils import TimeArray, SpatialArray, TimeSpatialArray


class Propeller(DynamicObject):

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
        args = {"obj_name": self.name, "ub": 20, "lb": -20}
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}n_prop [rps]", **args})
        args["var_type"] = "observation"
        self.var_info.append({"var_name": f"{self.o_tag}n_prop [rps]", **args})
        args["var_type"] = "external_state"
        self.var_info.append({"var_name": f"{self.e_tag}n_prop [rps]", **args})
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
        axes.set_ylim(-21, 21)
        # set y-axis labels
        axes.set_ylabel("$n_{\\mathrm{P}} \\ \\mathrm{(rps)}$")
        axes.set_xlabel("$t \\ \\mathrm{(s)}$")
        axes.legend()
        return fig, axes


class ControllablePitchPropeller(Propeller):

    def __init__(
        self,
        solve_method: str = "rk4",
        interp_method: str = "previous",
        K: npt.ArrayLike = [20, 20],
        T: npt.ArrayLike = [0.1, 0.1],
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
        self.K, self.T = np.array(K), np.array(T)

    def regist_variables(self):
        super().regist_variables()
        args = {"obj_name": self.name, "ub": np.deg2rad(20), "lb": -np.deg2rad(20)}
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}pitch_ang [rad]", **args})
        args["var_type"] = "observation"
        self.var_info.append({"var_name": f"{self.o_tag}pitch_ang [rad]", **args})
        args["var_type"] = "external_state"
        self.var_info.append({"var_name": f"{self.e_tag}pitch_ang [rad]", **args})
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
        kwargs = {"sharex": True, "figsize": (4.8, 2 * 1.2), "tight_layout": True}
        fig, axes = plt.subplots(2, 1, **kwargs)
        kwargs = {"color": "black", "lw": 0.5, "label": "Simulation"}
        axes[0].plot(t_seq, state_seq[:, 0], **kwargs)
        kwargs = {"color": "red", "lw": 0.5, "label": "Command"}
        axes[0].plot(t_seq, external_state_seq[:, 0], **kwargs)
        kwargs = {"color": "black", "lw": 0.5, "label": "Simulation"}
        axes[1].plot(t_seq, np.rad2deg(state_seq[:, 1]), **kwargs)
        kwargs = {"color": "red", "lw": 0.5, "label": "Command"}
        axes[1].plot(t_seq, np.rad2deg(external_state_seq[:, 1]), **kwargs)
        # set y-axis limits
        axes[0].set_ylim(-21, 21)
        axes[1].set_ylim(-21, 21)
        # set y-axis labels
        axes[0].set_ylabel("$n_{\\mathrm{P}} \\ \\mathrm{(rps)}$")
        axes[1].set_ylabel("$\\theta \\ \\mathrm{(deg.)}$")
        axes[-1].set_xlabel("$t \\ \\mathrm{(s)}$")
        axes[0].legend()
        return fig, axes
