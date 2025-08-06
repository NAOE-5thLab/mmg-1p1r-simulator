import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from ..object import StochasticDynamicObject
from ..utils import TimeArray, SpatialArray, TimeSpatialArray


class RandomWind(StochasticDynamicObject):

    def __init__(
        self,
        solve_method: str = "euler",
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

    def reset(
        self, state: npt.ArrayLike, seed: int | np.random.Generator = None
    ) -> None:
        super().reset(state, seed)
        u, xi = state
        assert u > 0.0, "Wind speed must be positive"
        self.u_bar, self.xi_bar = u, xi
        self.u_coeff = self.get_filtter_coeff_of_wind_speed(f_ref=0.5, u_10=u)
        self.u_alpha, self.u_beta, self.u_sigma = self.u_coeff
        self.xi_coeff = self.get_filtter_coeff_of_wind_direction(sigma_dir=2.3, u_10=u)
        self.xi_alpha, self.xi_beta, self.xi_sigma = self.xi_coeff

    def time_invariant_sde_rhs(
        self, state: SpatialArray, external_state: SpatialArray
    ) -> tuple[SpatialArray, SpatialArray]:
        u, xi = state
        u_mu = -self.u_alpha * (u - self.u_bar)
        xi_mu = -self.xi_alpha * np.sin(xi - self.xi_bar)
        return np.array([u_mu, xi_mu]), np.array([self.u_sigma, self.xi_sigma])

    def get_filtter_coeff_of_wind_speed(self, f_ref: float, u_10: float):
        ### Davenport and Hino ###
        Z = 15.0
        Kfriction = 0.001
        alpha_hino = 1.0 / 8.0
        m = 2
        u_bar = np.sqrt(6.0 * Kfriction * u_10**2)
        term1 = u_10 * alpha_hino / np.sqrt(Kfriction)
        term2 = (Z / 10.0) ** (2 * m * alpha_hino - 1.0)
        beta_hino = 1.169 * 1.0e-3 * term1 * term2
        ### Linear filter with Hino's spectrum ###
        #   Asymptotic value at f = 0
        Suw_H_0 = 0.2382 * u_bar**2 / beta_hino
        #   Asymptotic value at f = f_ref
        Suw_H_f_ref = Suw_H_0 * (1.0 + (f_ref / beta_hino) ** 2) ** (-5.0 / 6.0)
        #
        f_ref2 = f_ref**2
        pi2 = 2.0 * np.pi
        alpha2 = f_ref2 * pi2**2 * Suw_H_f_ref / (Suw_H_0 - Suw_H_f_ref)
        beta2 = Suw_H_0 * alpha2 / pi2
        #
        alpha = np.sqrt(alpha2)
        beta = np.sqrt(beta2)
        sigma = np.sqrt(2 * np.pi) * beta
        return alpha, beta, sigma

    def get_filtter_coeff_of_wind_direction(self, sigma_dir: float, u_10: float):
        sigma_dir2 = sigma_dir**2
        alpha = (sigma_dir2) * (u_10 ** (3.0 / 2.0)) / (2.0 * 32.0**2)
        beta2 = sigma_dir2 / (2 * np.pi)
        beta = np.sqrt(beta2) * np.pi / 180
        sigma = sigma_dir * np.pi / 180
        return alpha, beta, sigma

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
