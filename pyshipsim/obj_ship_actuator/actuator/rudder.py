import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from ...object import DynamicObject
from ...utils import TimeArray, SpatialArray, TimeSpatialArray


class Rudder(DynamicObject):

    def __init__(
        self,
        solve_method: str = "rk4",
        interp_method: str = "previous",
        K: float = np.deg2rad(20),
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
        args = {"obj_name": self.name, "ub": np.deg2rad(35), "lb": -np.deg2rad(35)}
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}delta_rudder [rad]", **args})
        args["var_type"] = "observation"
        self.var_info.append({"var_name": f"{self.o_tag}delta_rudder [rad]", **args})
        args["var_type"] = "external_state"
        self.var_info.append({"var_name": f"{self.e_tag}delta_rudder [rad]", **args})
        return self.var_info

    def time_invariant_ode_rhs(
        self, state: SpatialArray, external_state: SpatialArray
    ) -> SpatialArray:
        y, r = state, external_state
        dydt = np.clip((r - y) / self.T, -self.K, self.K)
        # dydt = self.K * np.sign(r - y)
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
        axes.plot(t_seq, np.rad2deg(state_seq), **kwargs)
        kwargs = {"color": "red", "lw": 0.5, "label": "Command"}
        axes.plot(t_seq, np.rad2deg(external_state_seq), **kwargs)
        # set y-axis limits
        axes.set_ylim(-40, 40)
        # set y-axis labels
        axes.set_ylabel("$\\delta \\ \\mathrm{(deg.)}$")
        axes.set_xlabel("$t \\ \\mathrm{(s)}$")
        axes.legend()
        return fig, axes


class VecTwinRudder(DynamicObject):

    def __init__(
        self,
        solve_method: str = "rk4",
        interp_method: str = "previous",
        K: npt.ArrayLike = [np.deg2rad(20), np.deg2rad(20)],
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
        self.var_info = []
        args = {"obj_name": self.name}
        args["ub"], args["lb"] = np.deg2rad(35), -np.deg2rad(105)
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}delta_rudder_p [rad]", **args})
        args["var_type"] = "observation"
        self.var_info.append({"var_name": f"{self.o_tag}delta_rudder_p [rad]", **args})
        args["var_type"] = "external_state"
        self.var_info.append({"var_name": f"{self.e_tag}delta_rudder_p [rad]", **args})
        args["ub"], args["lb"] = np.deg2rad(105), -np.deg2rad(35)
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}delta_rudder_s [rad]", **args})
        args["var_type"] = "observation"
        self.var_info.append({"var_name": f"{self.o_tag}delta_rudder_s [rad]", **args})
        args["var_type"] = "external_state"
        self.var_info.append({"var_name": f"{self.e_tag}delta_rudder_s [rad]", **args})
        return self.var_info

    def time_invariant_ode_rhs(
        self, state: SpatialArray, external_state: SpatialArray
    ) -> SpatialArray:
        y, r = state, external_state
        dydt = np.clip((r - y) / self.T, -self.K, self.K)
        # dydt = self.K * np.sign(r - y)
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
        for i, ax in enumerate(axes):
            kwargs = {"color": "black", "lw": 0.5, "label": "Simulation"}
            ax.plot(t_seq, np.rad2deg(state_seq[:, i]), **kwargs)
            kwargs = {"color": "red", "lw": 0.5, "label": "Command"}
            ax.plot(t_seq, np.rad2deg(external_state_seq[:, i]), **kwargs)
        # set y-axis limits
        axes[0].set_ylim(-110, 40)
        axes[1].set_ylim(-40, 110)
        # set y-axis labels
        axes[0].set_ylabel("$\\delta_{\\mathrm{P}} \\ \\mathrm{(deg.)}$")
        axes[1].set_ylabel("$\\delta_{\\mathrm{S}} \\ \\mathrm{(deg.)}$")
        axes[-1].set_xlabel("$t \\ \\mathrm{(s)}$")
        axes[0].legend()
        return fig, axes

        # def first_order_delay(self, y, r):
        #     dydt = (self.K * r - y) / self.T
        #     return dydt

        # def step(self, y, r):
        #     dydt = self.K * np.sign(r - y)
        #     return dydt

        # def step_with_slope(self, y, r):
        #     dydt = np.clip((r - y) / self.T, -self.K, self.K)
        #     return dydt
