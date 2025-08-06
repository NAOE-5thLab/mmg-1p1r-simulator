import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp as scipy_solve_ivp
from scipy.interpolate import interp1d

from .utils import TimeArray, SpatialArray, TimeSpatialArray, PolyArray


class SimulatedObject(object):
    def __init__(
        self,
        s_tag: str = None,
        o_tag: str = None,
        e_tag: str = None,
    ) -> None:
        self.name = f"{self.__class__.__name__}"
        fix_tag = lambda tag: "" if tag is None else f"{tag}_"
        self.s_tag = fix_tag(s_tag)
        self.o_tag = fix_tag(o_tag)
        self.e_tag = fix_tag(e_tag)


class StaticObject(SimulatedObject):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_polygons(self) -> list[tuple[PolyArray, bool]]:
        raise NotImplementedError


def solve_ivp(
    fun: callable,
    t_span: tuple[float, float],
    y0: SpatialArray,
    t_eval: TimeArray,
    method: str,
):
    if method == "Euler":
        state_seq = np.empty((len(t_eval), len(y0)))
        state_seq[0] = y0
        for ti in range(len(t_eval) - 1):
            dt = t_eval[ti + 1] - t_eval[ti]
            dxdt = fun(t_eval[ti], state_seq[ti])
            state_seq[ti + 1] = state_seq[ti] + dt * dxdt
        return state_seq
    elif method == "RK4":
        state_seq = np.empty((len(t_eval), len(y0)))
        state_seq[0] = y0
        for ti in range(len(t_eval) - 1):
            dt = t_eval[ti + 1] - t_eval[ti]
            k1 = fun(t_eval[ti], state_seq[ti])
            k2 = fun(t_eval[ti] + dt / 2, state_seq[ti] + dt / 2 * k1)
            k3 = fun(t_eval[ti] + dt / 2, state_seq[ti] + dt / 2 * k2)
            k4 = fun(t_eval[ti] + dt, state_seq[ti] + dt * k3)
            dxdt = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            state_seq[ti + 1] = state_seq[ti] + dt * dxdt
        return state_seq
    else:
        sol = scipy_solve_ivp(
            fun=fun,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
        )
        return sol.y.T


class DynamicObject(SimulatedObject):
    implemented_solve_methods = {
        "euler": "Euler",
        "rk4": "RK4",
        "rk45": "RK45",
        "dop8": "DOP853",
    }
    recommended_interp_methods = ["linear", "previous", "next"]

    def __init__(
        self,
        solve_method: str = "euler",
        interp_method: str = "linear",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert solve_method in self.implemented_solve_methods.keys()
        self.solve_method = self.implemented_solve_methods[solve_method]
        assert interp_method in self.recommended_interp_methods
        self.interp_method = interp_method

    def regist_variables(self) -> list[dict]:
        raise NotImplementedError

    def reset(
        self,
        state: npt.ArrayLike,
        seed: int | np.random.Generator = None,
    ) -> None:
        # get seed
        self.np_random = np.random.default_rng(seed=seed)
        # set initial state
        if isinstance(state, (float, int)):
            state = np.array([state])
        state = np.array(state)
        self.state_seq = np.array(state)[np.newaxis]
        self.observation_seq = self.observe_state_seq(state[np.newaxis])

    def step(
        self,
        t_seq: TimeArray,
        external_state_seq: TimeSpatialArray,
    ) -> SpatialArray:
        self.external_state_intep = interp1d(
            t_seq,
            external_state_seq.T,
            kind=self.interp_method,
            fill_value="extrapolate",
        )
        # Solve ODE
        state_seq = solve_ivp(
            fun=self.ode_rhs,
            t_span=(t_seq[0], t_seq[-1]),
            y0=self.state_seq[-1],
            t_eval=t_seq,
            method=self.solve_method,
        )
        # Update state
        self.state_seq = state_seq
        self.observation_seq = self.observe_state_seq(state_seq)
        return self.state_seq[-1]

    def ode_rhs(self, t: float, state: SpatialArray) -> SpatialArray:
        external_state = self.external_state_intep(t)
        return self.time_invariant_ode_rhs(state, external_state)

    def time_invariant_ode_rhs(
        self,
        state: SpatialArray,
        external_state: SpatialArray,
    ) -> SpatialArray:
        raise NotImplementedError

    def get_state(self, idx=-1) -> SpatialArray:
        return self.state_seq[idx]

    def get_state_seq(self) -> TimeSpatialArray:
        return self.state_seq

    def observe_state_seq(self, state_seq: TimeSpatialArray) -> SpatialArray:
        return np.array([self.observe_state(state) for state in state_seq])

    def observe_state(self, state: SpatialArray) -> SpatialArray:
        return state

    def get_observation(self, idx=-1) -> SpatialArray:
        return self.observation_seq[idx]

    def get_observation_seq(self) -> TimeSpatialArray:
        return self.observation_seq


def sample_ivp(
    fun: callable,
    t_span: tuple[float, float],
    y0: SpatialArray,
    t_eval: TimeArray,
    method="euler_maruyama",
    np_random: np.random.Generator | None = None,
):
    if np_random is None:
        np_random = np.random
    if np.any(t_eval < min(*t_span)) or np.any(t_eval > max(*t_span)):
        raise ValueError("Values in `t_eval` are not within `t_span`.")
    if method == "euler_maruyama":
        state_seq = np.empty((len(t_eval), len(y0)))
        state_seq[0] = y0
        for ti in range(len(t_eval) - 1):
            dt = t_eval[ti + 1] - t_eval[ti]
            dW = np.sqrt(dt) * np_random.normal(size=len(y0))
            mu, sigma = fun(t_eval[ti], state_seq[ti])
            dstate = mu * dt + sigma * dW
            state_seq[ti + 1] = state_seq[ti] + dstate
    else:
        raise NotImplementedError
    return state_seq


class StochasticDynamicObject(DynamicObject):
    implemented_solve_methods = {"euler": "euler_maruyama"}

    def __init__(
        self,
        solve_method: str = "euler",
        interp_method: str = "linear",
        **kwargs,
    ) -> None:
        super().__init__(solve_method, interp_method, **kwargs)

    def step(
        self, t_seq: SpatialArray, external_state_seq: SpatialArray
    ) -> SpatialArray:
        self.external_state_intep = interp1d(
            t_seq,
            external_state_seq.T,
            kind=self.interp_method,
            fill_value="extrapolate",
        )
        # Solve SDE
        state_seq = sample_ivp(
            fun=self.sde_rhs,
            t_span=(t_seq[0], t_seq[-1]),
            y0=self.state_seq[-1],
            t_eval=t_seq,
            method=self.solve_method,
            np_random=self.np_random,
        )
        # Update state
        self.state_seq = state_seq
        self.observation_seq = self.observe_state_seq(state_seq)
        return self.state_seq[-1]

    def sde_rhs(self, t: float, state: SpatialArray) -> SpatialArray:
        external_state = self.external_state_intep(t)
        return self.time_invariant_sde_rhs(state, external_state)

    def time_invariant_sde_rhs(
        self, state: SpatialArray, external_state: SpatialArray
    ) -> tuple[SpatialArray, SpatialArray]:
        raise NotImplementedError


# class SimulatedObject(object):

#     def __init__(self, tag: str = None) -> None:
#         self.tag = "" if tag is None else f"{tag}_"
#         self.tex_state_labels = {
#             # ship state
#             f"{self.tag}x_position_mid [m]": "$x_{0} \\ \\mathrm{(m)}$",
#             f"{self.tag}u_velo [m/s]": "$u \\ \\mathrm{(m/s)}$",
#             f"{self.tag}y_position_mid [m]": "$y_{0} \\ \\mathrm{(m)}$",
#             f"{self.tag}vm_velo [m/s]": "$v_{\\mathrm{m}} \\ \\mathrm{(m/s)}$",
#             f"{self.tag}psi [rad]": "$\\psi \\ \\mathrm{(deg.)}$",
#             f"{self.tag}r_angvelo [rad/s]": "$r \\ \\mathrm{(deg./s)}$",
#             # actuator state
#             f"{self.tag}delta_rudder [rad]": "$\\delta \\ \\mathrm{(deg.)}$",
#             f"{self.tag}delta_rudder_p [rad]": "$\\delta_{\\mathrm{P}} \\ \\mathrm{(deg.)}$",
#             f"{self.tag}delta_rudder_s [rad]": "$\\delta_{\\mathrm{S}} \\ \\mathrm{(deg.)}$",
#             f"{self.tag}n_prop [rps]": "$n_{\\mathrm{P}} \\ \\mathrm{(rps)}$",
#             f"{self.tag}n_bt [rps]": "$n_{\\mathrm{BT}} \\ \\mathrm{(rps)}$",
#             f"{self.tag}n_st [rps]": "$n_{\\mathrm{ST}} \\ \\mathrm{(rps)}$",
#             # wind state
#             f"{self.tag}true_wind_speed [m/s]": "$U_{\\mathrm{T}} \\ \\mathrm{(m/s)}$",
#             f"{self.tag}true_wind_direction [rad]": "$\\xi_{\\mathrm{T}} \\ \\mathrm{(deg.)}$",
#             # ship reference state
#             f"{self.tag}des_x_position_mid [m]": "$x_{0}^{\\mathrm{(des)}} \\ \\mathrm{(m)}$",
#             f"{self.tag}des_y_position_mid [m]": "$y_{0}^{\\mathrm{(des)}} \\ \\mathrm{(m)}$",
#             f"{self.tag}des_psi [rad]": "$\\psi^{\\mathrm{(des)}} \\ \\mathrm{(deg.)}$",
#             f"{self.tag}ref_x_position_mid [m]": "$x_{0}^{\\mathrm{(ref)}} \\ \\mathrm{(m)}$",
#             f"{self.tag}ref_y_position_mid [m]": "$y_{0}^{\\mathrm{(ref)}} \\ \\mathrm{(m)}$",
#             f"{self.tag}ref_psi [rad]": "$\\psi^{\\mathrm{(ref)}} \\ \\mathrm{(deg.)}$",
#             f"{self.tag}ref_x_position_mid_dot [m/s]": "$\\dot{x}_{0}^{\\mathrm{(ref)}} \\ \\mathrm{(m)}$",
#             f"{self.tag}ref_y_position_mid_dot [m/s]": "$\\dot{y}_{0}^{\\mathrm{(ref)}} \\ \\mathrm{(m)}$",
#             f"{self.tag}ref_psi_dot [rad/s]": "$\\dot{\\psi}^{\\mathrm{(ref)}} \\ \\mathrm{(deg.)}$",
#             f"{self.tag}ref_x_position_mid_dot_dot [m/s2]": "$\\ddot{x}_{0}^{\\mathrm{(ref)}} \\ \\mathrm{(m)}$",
#             f"{self.tag}ref_y_position_mid_dot_dot [m/s2]": "$\\ddot{y}_{0}^{\\mathrm{(ref)}} \\ \\mathrm{(m)}$",
#             f"{self.tag}ref_psi_dot_dot [rad/s2]": "$\\ddot{\\psi}^{\\mathrm{(ref)}} \\ \\mathrm{(deg.)}$",
#             f"{self.tag}ref_u_velo [m/s]": "$u^{\\mathrm{(ref)}} \\ \\mathrm{(m/s)}$",
#             f"{self.tag}ref_vm_velo [m/s]": "$v_{\\mathrm{m}}^{\\mathrm{(ref)}} \\ \\mathrm{(m/s)}$",
#             f"{self.tag}ref_r_angvelo [rad/s]": "$r^{\\mathrm{(ref)}} \\ \\mathrm{(deg./s)}$",
#             f"{self.tag}ref_u_dot [m/s2]": "$\\dot{u}^{\\mathrm{(ref)}} \\ \\mathrm{(m/s)}$",
#             f"{self.tag}ref_vm_dot [m/s2]": "$\\dot{v}_{\\mathrm{m}}^{\\mathrm{(ref)}} \\ \\mathrm{(m/s)}$",
#             f"{self.tag}ref_r_dot [rad/s2]": "$\\dot{r}^{\\mathrm{(ref)}} \\ \\mathrm{(deg./s)}$",
#         }
#         self.tex_observation_labels = {
#             key.replace(" ", "_hat "): value
#             for key, value in self.tex_state_labels.items()
#         }
#         #
#         self.STATE_NAME = []
#         self.STATE_UPPER_BOUND = []
#         self.STATE_LOWER_BOUND = []
#         self.EXTERNAL_STATE_NAME = []
#         self.EXTERNAL_STATE_UPPER_BOUND = []
#         self.EXTERNAL_STATE_LOWER_BOUND = []
#         self.OBSERVATION_NAME = []
#         self.OBSERVATION_UPPER_BOUND = []
#         self.OBSERVATION_LOWER_BOUND = []


# class StaticObject(SimulatedObject):

#     def __init__(self, tag: str = None) -> None:
#         super().__init__(tag)

#     def get_polygons(self) -> list[tuple[PolyArray, bool]]:
#         raise NotImplementedError


# class SimulatedObject(object):
#     implemented_solve_methods = ["euler", "rk4", "dop8"]
#     recommended_interp_methods = ["linear", "previous"]

#     def __init__(self, solve_method: str = "rk4", interp_method: str = "linear"):
#         assert solve_method in self.implemented_solve_methods
#         self.solve_method = solve_method
#         assert interp_method in self.recommended_interp_methods
#         self.interp_method = interp_method
#         self.logger = Logger()
#         #
#         self.STATE_NAME = []
#         self.STATE_UPPER_BOUND = []
#         self.STATE_LOWER_BOUND = []
#         self.EXTERNAL_STATE_NAME = []
#         self.EXTERNAL_STATE_UPPER_BOUND = []
#         self.EXTERNAL_STATE_LOWER_BOUND = []
#         self.OBSERVATION_NAME = []
#         self.OBSERVATION_UPPER_BOUND = []
#         self.OBSERVATION_LOWER_BOUND = []
#         self.setup()

#     def setup(self) -> None:
#         raise NotImplementedError

#     def reg_var(
#         self, name: str, ub: float = np.inf, lb: float = -np.inf, mode: str = "none"
#     ) -> None:
#         if mode == "state":
#             name_list = self.STATE_NAME
#             ub_list = self.STATE_UPPER_BOUND
#             lb_list = self.STATE_LOWER_BOUND
#         elif mode == "observation":
#             name_list = self.OBSERVATION_NAME
#             ub_list = self.OBSERVATION_UPPER_BOUND
#             lb_list = self.OBSERVATION_LOWER_BOUND
#         elif mode == "external_state":
#             name_list = self.EXTERNAL_STATE_NAME
#             ub_list = self.EXTERNAL_STATE_UPPER_BOUND
#             lb_list = self.EXTERNAL_STATE_LOWER_BOUND
#         else:
#             raise ValueError(
#                 "Mode must be either 'state', 'observation', or 'external_state'"
#             )

#         if name in name_list:
#             i = name_list.index(name)
#             name_list[i] = name
#             ub_list[i] = ub
#             lb_list[i] = lb
#         else:
#             name_list.append(name)
#             ub_list.append(ub)
#             lb_list.append(lb)

#     def reset(
#         self, state: npt.ArrayLike, seed: int | np.random.Generator = None
#     ) -> None:
#         self.np_random = get_np_random(seed=seed)
#         assert len(state) == len(self.STATE_NAME)
#         self.state_seq = np.array(state)[np.newaxis]
#         self.observation_seq = self.observe_state(state)[np.newaxis]
#         # logging
#         columns = self.STATE_NAME + self.OBSERVATION_NAME
#         self.logger.reset(columns)

#     def logging(self):
#         log = np.concatenate([self.state_seq, self.observation_seq], axis=-1)
#         for i in range(len(log)):
#             self.logger.add_row(log[i])

#     def log2df(self) -> pd.DataFrame:
#         return self.logger.get_df()

#     def odeint(
#         self, t_seq: TimeArray, external_state_seq: TimeSpatialArray
#     ) -> SpatialArray:
#         self.external_state_intep = interp1d(
#             t_seq,
#             external_state_seq.T,
#             kind=self.interp_method,
#             fill_value="extrapolate",
#         )
#         solve_method_convert = {
#             "euler": "Euler",
#             "rk4": "RK45",
#             "dop8": "DOP853",
#         }
#         state_seq = solve_ivp(
#             fun=self.ode_rhs,
#             t_span=(t_seq[0], t_seq[-1]),
#             y0=self.state_seq[-1],
#             t_eval=t_seq,
#             method=solve_method_convert[self.solve_method],
#         )
#         self.state_seq = state_seq
#         self.observation_seq = np.array(
#             [self.observe_state(state) for state in self.state_seq]
#         )
#         return self.state_seq[-1]

#     def ode_rhs(self, t: float, state: SpatialArray) -> SpatialArray:
#         external_state = self.external_state_intep(t)
#         return self.time_invariant_ode_rhs(state, external_state)

#     def time_invariant_ode_rhs(
#         self, state: SpatialArray, external_state: SpatialArray
#     ) -> SpatialArray:
#         return np.zeros_like(state)

#     def observe_state(self, state: SpatialArray) -> SpatialArray:
#         raise NotImplementedError

#     def create_polygons(self, state: SpatialArray) -> list[PolyArray]:
#         raise NotImplementedError

#     def get_state(self, idx=None) -> SpatialArray:
#         if idx is not None:
#             return self.state_seq[idx]
#         return self.state_seq[0]

#     def get_state_seq(self) -> TimeSpatialArray:
#         return self.state_seq

#     def get_observation(self, idx=None) -> SpatialArray:
#         if idx is not None:
#             return self.observation_seq[idx]
#         return self.observation_seq[0]

#     def get_observation_seq(self) -> TimeSpatialArray:
#         return self.observation_seq

#     def get_polygons(self, idx=None) -> list[PolyArray]:
#         if idx is not None:
#             return self.create_polygons(self.state_seq[idx])
#         return self.create_polygons(self.state_seq[-1])

#     def get_polygons_seq(self) -> list[list[PolyArray]]:
#         return [self.create_polygons(state) for state in self.state_seq]


# def sample_ivp(
#     fun: callable,
#     t_span: tuple[float, float],
#     y0: SpatialArray,
#     t_eval: TimeArray,
#     method="euler_maruyama",
#     np_random: np.random.Generator | None = None,
# ):
#     if np_random is None:
#         np_random = np.random
#     if np.any(t_eval < min(*t_span)) or np.any(t_eval > max(*t_span)):
#         raise ValueError("Values in `t_eval` are not within `t_span`.")
#     if method == "euler_maruyama":
#         state_seq = np.empty((len(t_eval), len(y0)))
#         state_seq[0] = y0
#         for ti in range(len(t_eval) - 1):
#             dt = t_eval[ti + 1] - t_eval[ti]
#             dW = np.sqrt(dt) * np_random.normal(size=len(y0))
#             mu, sigma = fun(t_eval[ti], state_seq[ti])
#             dstate = mu * dt + sigma * dW
#             state_seq[ti + 1] = state_seq[ti] + dstate
#     else:
#         raise NotImplementedError
#     return state_seq


# class SimulatedstochasticObject(SimulatedObject):
#     implemented_solve_methods = ["euler_maruyama"]

#     def odeint(
#         self, t_seq: SpatialArray, external_state_seq: SpatialArray
#     ) -> SpatialArray:
#         return self.sdeint(t_seq, external_state_seq)

#     def sdeint(
#         self, t_seq: SpatialArray, external_state_seq: SpatialArray
#     ) -> SpatialArray:
#         self.external_state_intep = interp1d(
#             t_seq,
#             external_state_seq,
#             kind=self.interp_method,
#             fill_value="extrapolate",
#         )
#         state_seq = sample_ivp(
#             fun=self.sde_rhs,
#             t_span=(t_seq[0], t_seq[-1]),
#             y0=self.state_seq[-1],
#             t_eval=t_seq,
#             method="euler_maruyama",
#             np_random=self.np_random,
#         )
#         self.state_seq = state_seq
#         self.observation_seq = np.array(
#             [self.observe_state(state) for state in self.state_seq]
#         )
#         return self.state_seq[-1]

#     def sde_rhs(self, t: float, state: SpatialArray) -> SpatialArray:
#         external_state = self.external_state_intep(t)
#         return self.time_invariant_ode_rhs(state, external_state)

#     def time_invariant_sde_rhs(
#         self, state: SpatialArray, external_state: SpatialArray
#     ) -> tuple[SpatialArray, SpatialArray]:
#         raise NotImplementedError


# class StaticObject(SimulatedObject):

#     def reset(
#         self, state: npt.ArrayLike, seed: int | np.random.Generator = None
#     ) -> None:
#         pass

#     def logging(self):
#         pass

#     def odeint(
#         self, t_seq: TimeArray, external_state_seq: TimeSpatialArray
#     ) -> SpatialArray:
#         pass

#     def create_polygons(self) -> list[PolyArray]:
#         raise NotImplementedError

#     def get_polygons(self, idx=None) -> list[PolyArray]:
#         return self.create_polygons()

#     def get_polygons_seq(self) -> list[list[PolyArray]]:
#         return [self.create_polygons(state) for state in self.state_seq]
