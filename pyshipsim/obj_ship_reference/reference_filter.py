import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from ..object import DynamicObject
from ..utils import TimeArray, SpatialArray, TimeSpatialArray
from ..utils import cmapline


def get_J(pose: npt.NDArray):
    psi = pose[2]
    Jinv = np.array(
        [
            [np.cos(psi), -np.sin(psi), 0.0],
            [np.sin(psi), np.cos(psi), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return Jinv


def get_Jinv(pose: npt.NDArray):
    return get_J(-pose)


def get_skew_matrix(omega: npt.NDArray):
    S = np.array(
        [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ]
    )
    return S


class ReferenceFilter(DynamicObject):

    def __init__(
        self,
        solve_method: str = "rk4",
        interp_method: str = "previous",
        s_tag: str = "des",
        o_tag: str = "des",
        e_tag: str = "ref",
        delta: npt.ArrayLike = None,  # Damping ratio
        omega: npt.ArrayLike = None,  # Natural frequency
        **kwargs,
    ):
        super().__init__(
            solve_method=solve_method,
            interp_method=interp_method,
            s_tag=s_tag,
            o_tag=o_tag,
            e_tag=e_tag,
            **kwargs,
        )
        #
        delta = np.ones(3) if delta is None else np.array(delta)
        omega = 0.03 * np.ones(3) if omega is None else np.array(omega)
        omega = np.array(omega)
        assert delta.ndim <= 2, "Damping ratio must be a 1D or 2D array"
        if delta.ndim == 1:
            delta = np.diag(delta)
        assert omega.ndim <= 2, "Natural frequency must be a 1D or 2D array"
        if omega.ndim == 1:
            omega = np.diag(omega)
        #
        self.D = delta  # Delta
        self.O = omega  # Omega
        self.Z = np.zeros((3, 3))
        self.I = np.identity(3)
        self.G = 2 * self.D + self.I
        self.O2 = self.O @ self.O
        self.O3 = self.O @ self.O2
        self.GO = self.G @ self.O
        self.GO2 = self.G @ self.O2

    def regist_variables(self):
        var_info = []
        args = {"obj_name": self.name, "ub": np.inf, "lb": -np.inf}
        args["var_type"] = "state"
        var_info.append({"var_name": f"{self.s_tag}x_position_mid [m]", **args})
        var_info.append({"var_name": f"{self.s_tag}y_position_mid [m]", **args})
        var_info.append({"var_name": f"{self.s_tag}psi [rad]", **args})
        var_info.append({"var_name": f"{self.s_tag}x_position_mid_dot [m/s]", **args})
        var_info.append({"var_name": f"{self.s_tag}y_position_mid_dot [m/s]", **args})
        var_info.append({"var_name": f"{self.s_tag}psi_dot [rad/s]", **args})
        var_info.append({"var_name": f"{self.s_tag}x_position_mid_ddot [m/s2]", **args})
        var_info.append({"var_name": f"{self.s_tag}y_position_mid_ddot [m/s2]", **args})
        var_info.append({"var_name": f"{self.s_tag}psi_ddot [rad/s2]", **args})
        args["var_type"] = "observation"
        var_info.append({"var_name": f"{self.o_tag}u_velo [m/s]", **args})
        var_info.append({"var_name": f"{self.o_tag}vm_velo [m/s]", **args})
        var_info.append({"var_name": f"{self.o_tag}r_angvelo [rad/s]", **args})
        var_info.append({"var_name": f"{self.o_tag}u_dot [m/s2]", **args})
        var_info.append({"var_name": f"{self.o_tag}vm_dot [m/s2]", **args})
        var_info.append({"var_name": f"{self.o_tag}r_dot [rad/s2]", **args})
        args["var_type"] = "external_state"
        var_info.append({"var_name": f"{self.e_tag}x_position_mid [m]", **args})
        var_info.append({"var_name": f"{self.e_tag}y_position_mid [m]", **args})
        var_info.append({"var_name": f"{self.e_tag}psi [rad]", **args})
        self.var_info = var_info
        #
        self.tex_state_labels = {
            f"{self.s_tag}x_position_mid [m]": "$x_{0}^{\\mathrm{(des)}} \\ \\mathrm{(m)}$",
            f"{self.s_tag}y_position_mid [m]": "$y_{0}^{\\mathrm{(des)}} \\ \\mathrm{(m)}$",
            f"{self.s_tag}psi [rad]": "$\\psi^{\\mathrm{(des)}} \\ \\mathrm{(deg.)}$",
            f"{self.s_tag}x_position_mid_dot [m/s]": "$\\dot{x}_{0}^{\\mathrm{(des)}} \\ \\mathrm{(m/s)}$",
            f"{self.s_tag}y_position_mid_dot [m/s]": "$\\dot{y}_{0}^{\\mathrm{(des)}} \\ \\mathrm{(m/s)}$",
            f"{self.s_tag}psi_dot [rad/s]": "$\\dot{\\psi}^{\\mathrm{(des)}} \\ \\mathrm{(deg./s)}$",
            f"{self.s_tag}x_position_mid_ddot [m/s2]": "$\\ddot{x}_{0}^{\\mathrm{(des)}} \\ \\mathrm{(m/s^2)}$",
            f"{self.s_tag}y_position_mid_ddot [m/s2]": "$\\ddot{y}_{0}^{\\mathrm{(des)}} \\ \\mathrm{(m/s^2)}$",
            f"{self.s_tag}psi_ddot [rad/s2]": "$\\ddot{\\psi}^{\\mathrm{(des)}} \\ \\mathrm{(deg./s^2)}$",
            f"{self.o_tag}u_velo [m/s]": "$u^{\\mathrm{(des)}} \\ \\mathrm{(m/s)}$",
            f"{self.o_tag}vm_velo [m/s]": "$v_{\\mathrm{m}}^{\\mathrm{(des)}} \\ \\mathrm{(m/s)}$",
            f"{self.o_tag}r_angvelo [rad/s]": "$r^{\\mathrm{(des)}} \\ \\mathrm{(deg./s)}$",
            f"{self.o_tag}u_dot [m/s2]": "$\\dot{u}^{\\mathrm{(des)}} \\ \\mathrm{(m/s^2)}$",
            f"{self.o_tag}vm_dot [m/s2]": "$\\dot{v}_{\\mathrm{m}}^{\\mathrm{(des)}} \\ \\mathrm{(m/s^2)}$",
            f"{self.o_tag}r_dot [rad/s2]": "$\\dot{r}^{\\mathrm{(des)}} \\ \\mathrm{(deg./s^2)}$",
            f"{self.e_tag}x_position_mid [m]": "$x_{0}^{\\mathrm{(ref)}} \\ \\mathrm{(m)}$",
            f"{self.e_tag}y_position_mid [m]": "$y_{0}^{\\mathrm{(ref)}} \\ \\mathrm{(m)}$",
            f"{self.e_tag}psi [rad]": "$\\psi^{\\mathrm{(ref)}} \\ \\mathrm{(deg.)}$",
        }
        return self.var_info

    def reset(self, x: npt.ArrayLike, seed: int | np.random.Generator = None) -> None:
        x = np.array(x)
        state = np.zeros(9)
        state[0:3] = x[[0, 2, 4]]
        state[3:6] = get_J(state[0:3]) @ x[[1, 3, 5]]
        super().reset(state, seed)

    def time_invariant_ode_rhs(
        self, state: SpatialArray, external_state: SpatialArray
    ) -> SpatialArray:
        pose = self.get_desired_pose(state)
        dot_pose = self.get_desired_dot_pose(state)
        ddot_pose = self.get_desired_ddot_pose(state)
        dddot_pose = (
            -self.O3 @ pose - self.GO2 @ dot_pose - self.GO @ ddot_pose
        ) + self.O3 @ external_state
        dot_state = np.hstack([dot_pose, ddot_pose, dddot_pose])
        return dot_state

    def observe_state(self, state: SpatialArray) -> SpatialArray:
        observation = np.empty(6)
        observation[0:3] = self.get_desired_velocity(state)
        observation[3:6] = self.get_desired_acceleration(state, velo=observation[0:3])
        return observation

    def get_desired_pose(self, state: SpatialArray) -> SpatialArray:
        return state[0:3]

    def get_desired_dot_pose(self, state: SpatialArray) -> SpatialArray:
        return state[3:6]

    def get_desired_ddot_pose(self, state: SpatialArray) -> SpatialArray:
        return state[6:9]

    def get_desired_velocity(self, state: SpatialArray) -> SpatialArray:
        return get_Jinv(state[0:3]) @ state[3:6]

    def get_desired_acceleration(
        self,
        state: SpatialArray,
        velo: SpatialArray = None,
    ) -> SpatialArray:
        velo = self.get_desired_velocity(state) if velo is None else velo
        S = get_skew_matrix(np.array([0, 0, state[6]]))
        Jinv = get_Jinv(state[0:3])
        return -S @ velo + Jinv @ state[6:9]

    def subplot_timeseries(
        self,
        t_seq: TimeArray,
        state_seq: TimeSpatialArray,
        observation_seq: TimeSpatialArray,
        external_state_seq: TimeSpatialArray,
    ) -> tuple[plt.Figure, plt.Axes]:
        s_info = [info for info in self.var_info if info["var_type"] == "state"]
        o_info = [info for info in self.var_info if info["var_type"] == "observation"]
        e_info = [
            info for info in self.var_info if info["var_type"] == "external_state"
        ]
        #
        kwargs = {"sharex": True, "figsize": (2 * 4.8, 9 * 1.2), "tight_layout": True}
        fig, axes = plt.subplots(9, 2, **kwargs)
        for i in range(3):
            s_name = s_info[i]["var_name"]
            e_name = e_info[i]["var_name"]
            if "[rad" in s_name:
                state_seq[:, i] = np.rad2deg(state_seq[:, i])
            if "[rad" in e_name:
                external_state_seq[:, i] = np.rad2deg(external_state_seq[:, i])
            kwargs = {"color": "black", "lw": 0.5, "label": "Desired"}
            axes[i, 0].plot(t_seq, state_seq[:, i], **kwargs)
            kwargs = {"color": "red", "lw": 0.5, "label": "Reference"}
            axes[i, 0].plot(t_seq, external_state_seq[:, i], **kwargs)
            y_label = self.tex_state_labels[self.var_info[i]["var_name"]]
            axes[i, 0].set_ylabel(y_label)
        for i in range(3, 9):
            s_name = s_info[i]["var_name"]
            if "[rad" in s_name:
                state_seq[:, i] = np.rad2deg(state_seq[:, i])
            kwargs = {"color": "black", "lw": 0.5, "label": "Desired"}
            axes[i, 0].plot(t_seq, state_seq[:, i], **kwargs)
            y_label = self.tex_state_labels[self.var_info[i]["var_name"]]
            axes[i, 0].set_ylabel(y_label)
        for i in range(6):
            o_name = o_info[i]["var_name"]
            if "[rad" in o_name:
                observation_seq[:, i] = np.rad2deg(observation_seq[:, i])
            kwargs = {"color": "black", "lw": 0.5, "label": "Desired"}
            axes[i, 1].plot(t_seq, observation_seq[:, i], **kwargs)
            y_label = self.tex_state_labels[self.var_info[i + 9]["var_name"]]
            axes[i, 1].set_ylabel(y_label)
        for i in range(6, 9):
            axes[i, 1].remove()
        axes[-1, 0].set_xlabel("$t \\ \\mathrm{(s)}$")
        axes[-1, 1].set_xlabel("$t \\ \\mathrm{(s)}$")
        axes[0, 0].legend()
        axes[0, 1].legend()
        return fig, axes

    def axes_x0y0(
        self,
        ax: plt.Axes,
        state_seq: TimeSpatialArray,
        observation_seq: TimeSpatialArray,
        scale: float = 1.0,
        plot_observation: bool = False,
    ) -> None:
        # trajectory
        s_info = [info for info in self.var_info if info["var_type"] == "state"]
        s_names = [info["var_name"] for info in s_info]
        x_id = s_names.index(f"{self.s_tag}x_position_mid [m]")
        y_id = s_names.index(f"{self.s_tag}y_position_mid [m]")
        # kwargs = {"color": "blue", "ls": "solid", "lw": 0.5}
        # ax.plot(state_seq[:, y_id], state_seq[:, x_id], **kwargs)
        kwargs = {"palette": "Blues", "ls": "solid", "lw": 0.5}
        ax = cmapline(ax, state_seq[:, y_id], state_seq[:, x_id], **kwargs)
