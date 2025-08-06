import numpy as np
import numpy.typing as npt

from ...object import DynamicObject
from ...utils import TimeArray, SpatialArray, TimeSpatialArray


class ConstantCommander(DynamicObject):

    def __init__(
        self,
        solve_method="euler",
        interp_method="previous",
        s_tag="cmd",
        **kwargs,
    ):
        super().__init__(
            solve_method=solve_method,
            interp_method=interp_method,
            s_tag=s_tag,
            **kwargs,
        )

    def reset(
        self, state: npt.ArrayLike, seed: int | np.random.Generator = None
    ) -> None:
        # get seed
        self.np_random = np.random.default_rng(seed=seed)
        super().reset(state, self.np_random)

    def step(
        self, t_seq: TimeArray, external_state_seq: TimeSpatialArray
    ) -> SpatialArray:
        state = self.state_seq[-1]
        # Update state
        self.state_seq = np.array([state] * len(t_seq))
        self.observation_seq = np.array(
            [self.observe_state(state) for state in self.state_seq]
        )
        return self.state_seq[-1]

    def observe_state(self, state: SpatialArray) -> SpatialArray:
        return state


class UniformCommander(DynamicObject):

    def __init__(
        self,
        update_act_steps: int = 1,
        solve_method="euler",
        interp_method="previous",
        s_tag="cmd",
        **kwargs,
    ):
        super().__init__(
            solve_method=solve_method,
            interp_method=interp_method,
            s_tag=s_tag,
            **kwargs,
        )
        self.update_act_steps = update_act_steps
        self.var_info = []

    def reset(
        self, state: npt.ArrayLike, seed: int | np.random.Generator = None
    ) -> None:
        self.np_random = np.random.default_rng(seed=seed)
        s_info = [info for info in self.var_info if info["var_type"] == "state"]
        self.lb = np.array([info["lb"] for info in s_info])
        self.ub = np.array([info["ub"] for info in s_info])
        if state is None:
            self.act_count = 0
            state = self.np_random.uniform(low=self.lb, high=self.ub)
        else:
            self.act_count = np.inf
        super().reset(state, self.np_random)

    def step(
        self, t_seq: TimeArray, external_state_seq: TimeSpatialArray
    ) -> SpatialArray:
        if self.act_count >= self.update_act_steps:
            self.act_count = 0
            state = self.np_random.uniform(low=self.lb, high=self.ub)
            # Update state
            self.state_seq = np.array([state] * len(t_seq))
        self.observation_seq = np.array(
            [self.observe_state(state) for state in self.state_seq]
        )
        self.act_count += 1
        return self.state_seq[-1]

    def observe_state(self, state: SpatialArray) -> SpatialArray:
        return state


class NormalCommander(DynamicObject):
    def __init__(
        self,
        mean: npt.ArrayLike = None,
        cov: npt.ArrayLike = None,
        update_act_steps: int = 1,
        solve_method="euler",
        interp_method="previous",
        s_tag="cmd",
        **kwargs,
    ):
        super().__init__(
            solve_method=solve_method,
            interp_method=interp_method,
            s_tag=s_tag,
            **kwargs,
        )
        self.update_act_steps = update_act_steps
        self.mean, self.cov = mean, cov
        self.var_info = []

    def reset(
        self, state: npt.ArrayLike, seed: int | np.random.Generator = None
    ) -> None:
        self.np_random = np.random.default_rng(seed=seed)
        s_info = [info for info in self.var_info if info["var_type"] == "state"]
        self.lb = np.array([info["lb"] for info in s_info])
        self.ub = np.array([info["ub"] for info in s_info])
        #
        if self.mean is None:
            self.mean = np.array(self.lb + self.ub) / 2
        self.mean = np.array(self.mean)
        if self.cov is None:
            self.cov = np.diag(np.array(self.ub - self.lb) / 4)
        self.cov = np.array(self.cov)
        if state is None:
            self.act_count = 0
            state = self.np_random.multivariate_normal(
                mean=self.mean, cov=self.cov
            ).clip(min=self.lb, max=self.ub)
        else:
            self.act_count = np.inf
        super().reset(state, self.np_random)

    def step(
        self, t_seq: TimeArray, external_state_seq: TimeSpatialArray
    ) -> SpatialArray:
        if self.act_count >= self.update_act_steps:
            self.act_count = 0
            state = self.np_random.multivariate_normal(
                mean=self.mean, cov=self.cov
            ).clip(min=self.lb, max=self.ub)
            # Update state
            self.state_seq = np.array([state] * len(t_seq))
        self.observation_seq = np.array(
            [self.observe_state(state) for state in self.state_seq]
        )
        self.act_count += 1
        return self.state_seq[-1]

    def observe_state(self, state: SpatialArray) -> SpatialArray:
        return state
