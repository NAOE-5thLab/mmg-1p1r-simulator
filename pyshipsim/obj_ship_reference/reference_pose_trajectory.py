import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from ..object import DynamicObject
from ..utils import TimeArray, SpatialArray, TimeSpatialArray


class ReferencePoseTrajectory(DynamicObject):

    def __init__(
        self,
        t_seq: npt.ArrayLike,
        pose_seq: npt.ArrayLike,
        interp_method: str = "previous",
        s_tag: str = "ref",
        **kwargs,
    ):
        super().__init__(
            interp_method=interp_method,
            s_tag=s_tag,
            **kwargs,
        )
        self.state_interp = interp1d(
            np.array(t_seq),
            np.array(pose_seq).T,
            kind=self.interp_method,
            fill_value="extrapolate",
        )

    def reset(
        self,
        state: npt.ArrayLike,
        seed: int | np.random.Generator = None,
    ) -> None:
        return super().reset(self.state_interp(0), seed)

    def regist_variables(self):
        self.var_info = []
        args = {"obj_name": self.name, "ub": np.inf, "lb": -np.inf}
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}x_position_mid [m]", **args})
        self.var_info.append({"var_name": f"{self.s_tag}y_position_mid [m]", **args})
        self.var_info.append({"var_name": f"{self.s_tag}psi [rad]", **args})
        #
        self.tex_state_labels = {
            f"{self.s_tag}x_position_mid [m]": "$x_{0} \\ \\mathrm{(m)}$",
            f"{self.s_tag}y_position_mid [m]": "$y_{0} \\ \\mathrm{(m)}$",
            f"{self.s_tag}psi [rad]": "$\\psi \\ \\mathrm{(deg.)}$",
        }
        return self.var_info

    def step(
        self, t_seq: TimeArray, external_state_seq: TimeSpatialArray
    ) -> SpatialArray:
        # Update state
        self.state_seq, self.observation_seq = [], []
        for t in t_seq:
            state = self.state_interp(t)
            self.state_seq.append(state)
            self.observation_seq.append(self.observe_state(state))
        self.state_seq = np.array(self.state_seq)
        self.observation_seq = np.array(self.observation_seq)
        return self.state_seq[-1]

    def subplot_timeseries(
        self,
        t_seq: TimeArray,
        state_seq: TimeSpatialArray,
        observation_seq: TimeSpatialArray,
        external_state_seq: TimeSpatialArray,
    ) -> tuple[plt.Figure, plt.Axes]:
        s_info = [info for info in self.var_info if info["var_type"] == "state"]
        kwargs = {"sharex": True, "figsize": (4.8, 3 * 1.2), "tight_layout": True}
        fig, axes = plt.subplots(3, 1, **kwargs)
        for i, ax in enumerate(axes):
            s_name = s_info[i]["var_name"]
            if "[rad]" in s_name or "[rad/s]" in s_name:
                state_seq[:, i] = np.rad2deg(state_seq[:, i])
            kwargs = {"color": "black", "lw": 0.5, "label": "Simulation"}
            ax.plot(t_seq, state_seq[:, i], **kwargs)
            y_label = self.tex_state_labels[s_name]
            ax.set_ylabel(y_label)
        axes[-1].set_xlabel("$t \\ \\mathrm{(s)}$")
        axes[0].legend()
        return fig, axes
