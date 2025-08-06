import numpy as np
import matplotlib.pyplot as plt

from .utils import simple_ship_poly, detail_ship_poly, rectangle_ship_poly
from ..object import DynamicObject
from ..utils import TimeArray, SpatialArray, TimeSpatialArray, PolyArray
from ..utils import cmapline


class SurfaceShip(DynamicObject):
    implemented_polygon_types = ["simple_ship", "detail_ship", "rectangle_ship"]

    def __init__(
        self,
        solve_method: str = "rk4",
        interp_method: str = "linear",
        polygon_type: str = "simple_ship",
        o_tag: str = "hat",
        **kwargs,
    ):
        super().__init__(
            solve_method=solve_method,
            interp_method=interp_method,
            o_tag=o_tag,
            **kwargs,
        )
        assert (
            polygon_type in self.implemented_polygon_types
        ), f"Invalid polygon type: {polygon_type}"
        self.polygon_type = polygon_type
        # state
        self.L, self.B = 3.0, 0.5

    def regist_variables(self):
        self.var_info = []
        args = {"obj_name": self.name, "ub": np.inf, "lb": -np.inf}
        args["var_type"] = "state"
        self.var_info.append({"var_name": f"{self.s_tag}x_position_mid [m]", **args})
        self.var_info.append({"var_name": f"{self.s_tag}u_velo [m/s]", **args})
        self.var_info.append({"var_name": f"{self.s_tag}y_position_mid [m]", **args})
        self.var_info.append({"var_name": f"{self.s_tag}vm_velo [m/s]", **args})
        self.var_info.append({"var_name": f"{self.s_tag}psi [rad]", **args})
        self.var_info.append({"var_name": f"{self.s_tag}r_angvelo [rad/s]", **args})
        args["var_type"] = "observation"
        self.var_info.append({"var_name": f"{self.o_tag}x_position_mid [m]", **args})
        self.var_info.append({"var_name": f"{self.o_tag}u_velo [m/s]", **args})
        self.var_info.append({"var_name": f"{self.o_tag}y_position_mid [m]", **args})
        self.var_info.append({"var_name": f"{self.o_tag}vm_velo [m/s]", **args})
        self.var_info.append({"var_name": f"{self.o_tag}psi [rad]", **args})
        self.var_info.append({"var_name": f"{self.o_tag}r_angvelo [rad/s]", **args})
        #
        self.tex_state_labels = {
            f"{self.s_tag}x_position_mid [m]": "$x_{0} \\ \\mathrm{(m)}$",
            f"{self.s_tag}u_velo [m/s]": "$u \\ \\mathrm{(m/s)}$",
            f"{self.s_tag}y_position_mid [m]": "$y_{0} \\ \\mathrm{(m)}$",
            f"{self.s_tag}vm_velo [m/s]": "$v_{\\mathrm{m}} \\ \\mathrm{(m/s)}$",
            f"{self.s_tag}psi [rad]": "$\\psi \\ \\mathrm{(deg.)}$",
            f"{self.s_tag}r_angvelo [rad/s]": "$r \\ \\mathrm{(deg./s)}$",
        }
        return self.var_info

    def get_polygons(self, state: SpatialArray) -> list[tuple[PolyArray, bool]]:
        if self.polygon_type == "simple_ship":
            ship_poly = simple_ship_poly(state, self.L, self.B)
        elif self.polygon_type == "detail_ship":
            ship_poly = detail_ship_poly(state, self.L, self.B)
        elif self.polygon_type == "rectangle_ship":
            ship_poly = rectangle_ship_poly(state, self.L, self.B)
        return [[ship_poly, False]]

    def axes_x0y0(
        self,
        ax: plt.Axes,
        state_seq: TimeSpatialArray,
        observation_seq: TimeSpatialArray,
        scale: float = 1.0,
        plot_observation: bool = True,
    ) -> None:
        # trajectory
        s_info = [info for info in self.var_info if info["var_type"] == "state"]
        s_names = [info["var_name"] for info in s_info]
        x_id = s_names.index(f"{self.s_tag}x_position_mid [m]")
        y_id = s_names.index(f"{self.s_tag}y_position_mid [m]")
        # kwargs = {"color": "black", "ls": "solid", "lw": 0.5}
        # ax.plot(state_seq[:, y_id], state_seq[:, x_id], **kwargs)
        kwargs = {"palette": "Greys", "ls": "solid", "lw": 0.5}
        ax = cmapline(ax, state_seq[:, y_id], state_seq[:, x_id], **kwargs)
        if plot_observation:
            o_info = [i for i in self.var_info if i["var_type"] == "observation"]
            o_names = [info["var_name"] for info in o_info]
            xh_id = o_names.index(f"{self.o_tag}x_position_mid [m]")
            yh_id = o_names.index(f"{self.o_tag}y_position_mid [m]")
            # kwargs = {"color": "red", "ls": "solid", "lw": 0.5}
            # ax.plot(observation_seq[:, yh_id], observation_seq[:, xh_id], **kwargs)
            kwargs = {"palette": "Reds", "ls": "solid", "lw": 0.5}
            ax = cmapline(ax, state_seq[:, y_id], state_seq[:, x_id], **kwargs)
        # polygons
        for i in range(len(state_seq) - 1, 0, -100):
            ship_poly = detail_ship_poly(state_seq[i, :], self.L, self.B)
            ship_poly_over_L = ship_poly[:, [1, 0]] / scale
            kwargs = {"fill": False, "lw": 0.3, "ec": "black"}
            polygon = plt.Polygon(ship_poly_over_L, **kwargs)
            ax.add_patch(polygon)
            if plot_observation:
                ship_poly = detail_ship_poly(observation_seq[i, :], self.L, self.B)
                ship_poly_over_L = ship_poly[:, [1, 0]] / scale
                kwargs = {"fill": False, "lw": 0.3, "ec": "red"}
                polygon = plt.Polygon(ship_poly_over_L, **kwargs)
                ax.add_patch(polygon)

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
        if len(e_info) > 0:
            kwargs = {"sharex": True, "figsize": (9.6, 7.2), "tight_layout": True}
            fig, axes = plt.subplots(max(6, len(e_info)), 2, **kwargs)
            # state
            for i in range(6):
                s_name = s_info[i]["var_name"]
                o_name = o_info[i]["var_name"]
                lb = s_info[i]["lb"]
                ub = s_info[i]["ub"]
                if "[rad]" in s_name or "[rad/s]" in s_name:
                    state_seq[:, i] = np.rad2deg(state_seq[:, i])
                    lb, ub = np.rad2deg(lb), np.rad2deg(ub)
                if "[rad]" in o_name or "[rad/s]" in o_name:
                    observation_seq[:, i] = np.rad2deg(observation_seq[:, i])
                # simulation
                kwargs = {"color": "black", "lw": 0.5, "label": "Simulation"}
                axes[i, 0].plot(t_seq, state_seq[:, i], **kwargs)
                # observation
                kwargs = {"color": "red", "lw": 0.5, "label": "Observation"}
                axes[i, 0].plot(t_seq, observation_seq[:, i], **kwargs)
                # set y-axis limits
                if not np.isinf(lb) and not np.isinf(ub):
                    lb_ = (lb + ub) * 0.5 - (ub - lb) * 0.6
                    ub_ = (lb + ub) * 0.5 + (ub - lb) * 0.6
                    axes[i, 0].set_ylim(lb_, ub_)
                # set y-axis labels
                axes[i, 0].set_ylabel(self.tex_state_labels[s_name])
            axes[-1, 0].set_xlabel("$t \\ \\mathrm{(s)}$")
            axes[0, 0].legend()
            # external state
            for i in range(len(e_info)):
                e_name = e_info[i]["var_name"]
                lb = e_info[i]["lb"]
                ub = e_info[i]["ub"]
                if "[rad]" in e_name or "[rad/s]" in e_name:
                    external_state_seq[:, i] = np.rad2deg(external_state_seq[:, i])
                if "true_wind_direction [rad]" in e_name:
                    external_state_seq[:, i] %= 360
                if "[rad]" in e_name or "[rad/s]" in e_name:
                    lb, ub = np.rad2deg(lb), np.rad2deg(ub)
                if not np.isinf(lb) and not np.isinf(ub):
                    lb_ = (lb + ub) * 0.5 - (ub - lb) * 0.6
                    ub_ = (lb + ub) * 0.5 + (ub - lb) * 0.6
                    axes[i, 1].set_ylim(lb_, ub_)
                # external state
                kwargs = {"color": "black", "lw": 0.5, "label": "Simulation"}
                axes[i, 1].plot(t_seq, external_state_seq[:, i], **kwargs)
                # set y-axis labels
                axes[i, 1].set_ylabel(self.tex_state_labels[e_name])
            axes[-1, 1].set_xlabel("$t \\ \\mathrm{(s)}$")
        else:
            kwargs = {"sharex": True, "figsize": (4.8, 7.2), "tight_layout": True}
            fig, axes = plt.subplots(6, 1, **kwargs)
            # state
            for i in range(6):
                s_name = s_info[i]["var_name"]
                o_name = o_info[i]["var_name"]
                lb = s_info[i]["lb"]
                ub = s_info[i]["ub"]
                if "[rad]" in s_name or "[rad/s]" in s_name:
                    state_seq[:, i] = np.rad2deg(state_seq[:, i])
                    lb, ub = np.rad2deg(lb), np.rad2deg(ub)
                if "[rad]" in o_name or "[rad/s]" in o_name:
                    observation_seq[:, i] = np.rad2deg(observation_seq[:, i])
                # simulation
                kwargs = {"color": "black", "lw": 0.5, "label": "Simulation"}
                axes[i].plot(t_seq, state_seq[:, i], **kwargs)
                # observation
                kwargs = {"color": "red", "lw": 0.5, "label": "Observation"}
                axes[i].plot(t_seq, observation_seq[:, i], **kwargs)
                # set y-axis limits
                if not np.isinf(lb) and not np.isinf(ub):
                    lb_ = (lb + ub) * 0.5 - (ub - lb) * 0.6
                    ub_ = (lb + ub) * 0.5 + (ub - lb) * 0.6
                    axes[i].set_ylim(lb_, ub_)
                # set y-axis labels
                axes[i].set_ylabel(self.tex_state_labels[s_name])
            axes[-1].set_xlabel("$t \\ \\mathrm{(s)}$")
            axes[0].legend()
        return fig, axes
