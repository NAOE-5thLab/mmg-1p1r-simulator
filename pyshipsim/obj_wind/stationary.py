import numpy as np
import matplotlib.pyplot as plt

from ..object import DynamicObject
from ..utils import TimeArray, SpatialArray, TimeSpatialArray


class StationaryWind(DynamicObject):

    def __init__(
        self,
        solve_method: str = "rk4",
        o_tag: str = "hat",
        **kwargs,
    ):
        super().__init__(
            solve_method=solve_method,
            o_tag=o_tag,
            **kwargs,
        )

    def regist_variables(self):
        var_info = []
        args = {"obj_name": self.name}
        args["ub"], args["lb"] = np.inf, 0.0
        args["var_type"] = "state"
        var_info.append({"var_name": f"{self.s_tag}true_wind_speed [m/s]", **args})
        args["var_type"] = "observation"
        var_info.append({"var_name": f"{self.o_tag}true_wind_speed [m/s]", **args})
        args["ub"], args["lb"] = 2 * np.pi, 0.0
        args["var_type"] = "state"
        var_info.append({"var_name": f"{self.s_tag}true_wind_direction [rad]", **args})
        args["var_type"] = "observation"
        var_info.append({"var_name": f"{self.o_tag}true_wind_direction [rad]", **args})
        self.var_info = var_info
        #
        self.tex_state_labels = {
            f"{self.s_tag}true_wind_speed [m/s]": "$U_{\\mathrm{T}} \\ \\mathrm{(m/s)}$",
            f"{self.s_tag}true_wind_direction [rad]": "$\\xi_{\\mathrm{T}} \\ \\mathrm{(deg.)}$",
        }
        return self.var_info

    def step(
        self, t_seq: TimeArray, external_state_seq: TimeSpatialArray
    ) -> SpatialArray:
        state = self.state_seq[-1]
        # Update state
        self.state_seq = np.array([state] * len(t_seq))
        self.observation_seq = self.observe_state_seq(self.state_seq)
        return self.state_seq[-1]

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
    ):
        s_info = [info for info in self.var_info if info["var_type"] == "state"]
        kwargs = {"sharex": True, "figsize": (4.8, 2 * 1.2), "tight_layout": True}
        fig, axes = plt.subplots(2, 1, **kwargs)
        for i, ax in enumerate(axes):
            s_name = s_info[i]["var_name"]
            lb = s_info[i]["lb"]
            ub = s_info[i]["ub"]
            if "[rad]" in s_name or "[rad/s]" in s_name:
                state_seq[:, i] = np.rad2deg(state_seq[:, i]) % 360
                observation_seq[:, i] = np.rad2deg(observation_seq[:, i]) % 360
                lb, ub = np.rad2deg(lb), np.rad2deg(ub)
            kwargs = {"color": "black", "lw": 0.5, "label": "Simulation"}
            ax.plot(t_seq, state_seq[:, i], **kwargs)
            kwargs = {"color": "red", "lw": 0.5, "label": "Observation"}
            ax.plot(t_seq, observation_seq[:, i], **kwargs)
            # set y-axis limits
            if not np.isinf(lb) and not np.isinf(ub):
                lb_ = (lb + ub) * 0.5 - (ub - lb) * 0.6
                ub_ = (lb + ub) * 0.5 + (ub - lb) * 0.6
                ax.set_ylim(lb_, ub_)
            # set y-axis labels
            ax.set_ylabel(self.tex_state_labels[s_name])
        axes[-1].set_xlabel("$t \\ \\mathrm{(s)}$")
        axes[0].legend()
        return fig, axes
