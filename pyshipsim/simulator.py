import os
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt

from .collision import *
from .object import *
from .obj_ship_mmg import SurfaceShip
from .obj_obstacle import Harbor
from .utils import SpatialArray, TimeSpatialArray


class VariablesInfo(object):
    def __init__(self):
        self.var_names = []
        self.var_upper_bounds = []
        self.var_lower_bounds = []
        self.var_for_obj = {"state": {}, "external_state": {}, "observation": {}}
        self.var_for_all = {"state": {}, "external_state": {}, "observation": {}}

    def regist_variable(
        self,
        obj_name: str,
        var_type: str,
        var_name: str,
        ub: float = np.inf,
        lb: float = -np.inf,
    ):
        # regist var
        if var_name in self.var_names:
            idx = self.var_names.index(var_name)
            self.var_upper_bounds[idx] = ub
            self.var_lower_bounds[idx] = lb
        else:
            self.var_names.append(var_name)
            self.var_upper_bounds.append(ub)
            self.var_lower_bounds.append(lb)
        idx = self.var_names.index(var_name)
        # regist relation with obj
        if obj_name in self.var_for_obj[var_type]:
            self.var_for_obj[var_type][obj_name][var_name] = idx
        else:
            self.var_for_obj[var_type][obj_name] = {var_name: idx}

    def update_var_for_all(self) -> dict:
        for var_name_idx in self.var_for_obj["state"].values():
            self.var_for_all["state"].update(var_name_idx)
        for var_name_idx in self.var_for_obj["external_state"].values():
            for var_name, var_idx in var_name_idx.items():
                if var_name in self.var_for_all["state"]:
                    continue
                self.var_for_all["external_state"][var_name] = var_idx
        for var_name_idx in self.var_for_obj["observation"].values():
            self.var_for_all["observation"].update(var_name_idx)

    def get_index(self, var_name: str) -> int | None:
        try:
            return self.var_names.index(var_name)
        except ValueError:
            return None

    def get_state_index_for_all(self) -> dict[str, int]:
        return self.var_for_all["state"]

    def get_external_state_index_for_all(self) -> dict[str, int]:
        return self.var_for_all["external_state"]

    def get_observation_index_for_all(self) -> dict[str, int]:
        return self.var_for_all["observation"]

    def get_state_index(self, obj_name: str) -> dict[str, int]:
        return self.var_for_obj["state"].get(obj_name, {})

    def get_external_state_index(self, obj_name: str) -> dict[str, int]:
        return self.var_for_obj["external_state"].get(obj_name, {})

    def get_observation_index(self, obj_name: str) -> dict[str, int]:
        return self.var_for_obj["observation"].get(obj_name, {})

    def get_state_upper_bounds(self, obj_name: str) -> dict[str, int]:
        idx_dict = self.get_state_index(obj_name)
        return {name: self.var_upper_bounds[idx] for name, idx in idx_dict.items()}

    def get_state_lower_bounds(self, obj_name: str) -> dict[str, int]:
        idx_dict = self.get_state_index(obj_name)
        return {name: self.var_lower_bounds[idx] for name, idx in idx_dict.items()}

    def get_external_state_upper_bounds(self, obj_name: str) -> dict[str, int]:
        idx_dict = self.get_external_state_index(obj_name)
        return {name: self.var_upper_bounds[idx] for name, idx in idx_dict.items()}

    def get_external_state_lower_bounds(self, obj_name: str) -> dict[str, int]:
        idx_dict = self.get_external_state_index(obj_name)
        return {name: self.var_lower_bounds[idx] for name, idx in idx_dict.items()}

    def get_observation_upper_bounds(self, obj_name: str) -> dict[str, int]:
        idx_dict = self.get_observation_index(obj_name)
        return {name: self.var_upper_bounds[idx] for name, idx in idx_dict.items()}

    def get_observation_lower_bounds(self, obj_name: str) -> dict[str, int]:
        idx_dict = self.get_observation_index(obj_name)
        return {name: self.var_lower_bounds[idx] for name, idx in idx_dict.items()}

    def reset_histroy(self):
        self.his = []

    def add_histroy(self, data: npt.NDArray):
        assert data.shape[1] == len(self.var_names)
        self.his.append(data)

    def get_df(self) -> pd.DataFrame:
        his = np.concatenate(self.his, axis=0)
        return pd.DataFrame(his, columns=self.var_names)

    def to_csv(self, path: str = "./test.csv") -> None:
        if not path.endswith(".csv"):
            path = f"{path}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.get_df().to_csv(path)


class Simulator(object):
    implemented_collide_ckecker = ["enclosing_point", "segments_intersect", "strict"]

    def __init__(
        self,
        *simobjs: SimulatedObject
        | StaticObject
        | DynamicObject
        | StochasticDynamicObject
        | SurfaceShip
        | Harbor,
        dt_act: float = 1.0,
        dt_sim: float = 0.1,
        collide_checker_type: str = None,
    ):
        # Objects
        self.simobjs = {simobj.name: simobj for simobj in simobjs}
        # Time-step
        assert abs(dt_act - round(dt_act / dt_sim) * dt_sim) < 1e-6
        self.dt_act = dt_act
        self.dt_sim = dt_sim
        # Collision checker
        if collide_checker_type == "enclosing_point":
            self.collide_checker = EnclosingPointCollisionChecker(
                *self.simobjs.values()
            )
        elif collide_checker_type == "segments_intersect":
            self.collide_checker = SegmentsIntersectCollisionChecker(
                *self.simobjs.values()
            )
        elif collide_checker_type == "strict":
            self.collide_checker = StrictCollisionChecker(*self.simobjs.values())
        elif collide_checker_type is not None:
            raise ValueError(f"Unknown collision checker: {collide_checker_type}")
        # Variables
        self.var_info = VariablesInfo()
        kwargs = {"obj_name": "self", "var_type": "state"}
        self.var_info.regist_variable(var_name="t [s]", ub=np.inf, lb=0, **kwargs)
        self.var_info.regist_variable(var_name="collide", ub=1, lb=0, **kwargs)
        self.var_info.regist_variable(var_name="terminated", ub=1, lb=0, **kwargs)
        for obj_name, simobj in self.simobjs.items():
            if hasattr(simobj, "regist_variables"):
                var_infos = simobj.regist_variables()
                for var_info in var_infos:
                    self.var_info.regist_variable(**var_info)
        self.var_info.update_var_for_all()
        # Order simobjs based on their external state dependencies
        org_obj_names, ord_obj_names = list(self.simobjs.keys()), []
        simed_state_name = set(self.var_info.get_external_state_index_for_all().keys())
        while org_obj_names:
            for obj_name in org_obj_names[:]:
                simobj = self.simobjs[obj_name]
                s_name = self.var_info.get_state_index(obj_name).keys()
                e_name = self.var_info.get_external_state_index(obj_name).keys()
                if all(name in simed_state_name for name in e_name):
                    simed_state_name.update(s_name)
                    ord_obj_names.append(obj_name)
                    org_obj_names.remove(obj_name)
                elif not e_name:
                    ord_obj_names.append(obj_name)
                    org_obj_names.remove(obj_name)
        self.ord_obj_names = ord_obj_names

    def reset(self, init_states: dict, seed: int | np.random.Generator = None):
        # get seed
        self.np_random = np.random.default_rng(seed=seed)
        # get idx
        self.state_idx, self.external_state_idx, self.observation_idx = {}, {}, {}
        self.state_idx["self"] = list(self.var_info.get_state_index("self").values())
        self.state_idx["all"] = list(self.var_info.get_state_index_for_all().values())
        self.external_state_idx["all"] = list(
            self.var_info.get_external_state_index_for_all().values()
        )
        self.observation_idx["all"] = list(
            self.var_info.get_observation_index_for_all().values()
        )
        for obj_name, simobj in self.simobjs.items():
            self.state_idx[obj_name] = list(
                self.var_info.get_state_index(obj_name).values()
            )
            self.external_state_idx[obj_name] = list(
                self.var_info.get_external_state_index(obj_name).values()
            )
            self.observation_idx[obj_name] = list(
                self.var_info.get_observation_index(obj_name).values()
            )
        # initiliaze state
        var_seq = np.empty((1, len(self.var_info.var_names)))
        var_seq[0, self.state_idx["self"]] = [0.0, False, False]
        for obj_name in self.ord_obj_names:
            simobj = self.simobjs[obj_name]
            if hasattr(simobj, "reset"):
                s_idx = self.state_idx[obj_name]
                o_idx = self.observation_idx[obj_name]
                # Update state
                simobj.reset(init_states.get(obj_name, None), seed=self.np_random)
                var_seq[:, s_idx] = simobj.get_state_seq()
                if len(o_idx) > 0:
                    var_seq[:, o_idx] = simobj.get_observation_seq()
        # Update and logging state
        self.var_seq = var_seq
        self.var_info.reset_histroy()
        self.var_info.add_histroy(var_seq)
        return self.get_observation()

    def step(
        self, external_state: npt.ArrayLike = np.zeros(0)
    ) -> tuple[SpatialArray, bool, dict]:
        var = self.var_seq[-1]
        t, _, terminated = var[self.state_idx["self"]]
        # external_state
        external_state = np.array(external_state)
        if external_state.ndim == 0:
            external_state = np.array([external_state])
        # Solve for the next state
        t_seq = np.arange(t, t + self.dt_act + self.dt_sim, self.dt_sim)
        var_seq = np.empty((len(t_seq), len(self.var_info.var_names)))
        var_seq[:, self.external_state_idx["all"]] = external_state
        # Update state of each simobj
        for obj_name in self.ord_obj_names:
            simobj = self.simobjs[obj_name]
            if hasattr(simobj, "step"):
                s_idx = self.state_idx[obj_name]
                e_idx = self.external_state_idx[obj_name]
                o_idx = self.observation_idx[obj_name]
                # Update state
                simobj.step(t_seq, var_seq[:, e_idx])
                var_seq[:, s_idx] = simobj.get_state_seq()
                if len(o_idx) > 0:
                    var_seq[:, o_idx] = simobj.get_observation_seq()
        # Check collision
        if hasattr(self, "collide_checker"):
            collide_seq = self.collide_checker.check_seq(t_seq)
            any_collide_seq = [np.any(collide_seq[: i + 1]) for i in range(len(t_seq))]
            terminated_seq = [terminated or collide for collide in any_collide_seq]
        else:
            collide_seq = [False] * len(t_seq)
            terminated_seq = [terminated] * len(t_seq)
        var_seq[:, self.state_idx["self"]] = np.stack(
            [t_seq, np.array(collide_seq), np.array(terminated_seq)], axis=1
        )
        # Update and logging state
        self.var_seq = var_seq
        self.var_info.add_histroy(var_seq[1:])
        # Return
        info = {"state": self.get_state(), "action": external_state}
        return self.get_observation(), terminated, info

    def get_info(self):
        msg = "------------------ Simulator Info ------------------\n"
        msg += "Given objects:\n"
        for obj_name, simobj in self.simobjs.items():
            msg += f"    {obj_name}:\n"
            msg += f"        State variables\n"
            s_names = self.var_info.get_state_index(obj_name).keys()
            s_ubs = self.var_info.get_state_upper_bounds(obj_name).values()
            s_lbs = self.var_info.get_state_lower_bounds(obj_name).values()
            for s_name, s_ub, s_lb in zip(s_names, s_ubs, s_lbs):
                msg += f"            {s_name}: [{s_lb}, {s_ub}]\n"
            msg += f"        External State variables\n"
            e_names = self.var_info.get_external_state_index(obj_name).keys()
            e_ubs = self.var_info.get_external_state_upper_bounds(obj_name).values()
            e_lbs = self.var_info.get_external_state_lower_bounds(obj_name).values()
            for e_name, e_ub, e_lb in zip(e_names, e_ubs, e_lbs):
                msg += f"            {e_name}: [{e_lb}, {e_ub}]\n"
            msg += f"        Observation variables\n"
            o_names = self.var_info.get_observation_index(obj_name).keys()
            o_ubs = self.var_info.get_observation_upper_bounds(obj_name).values()
            o_lbs = self.var_info.get_observation_lower_bounds(obj_name).values()
            for o_name, o_ub, o_lb in zip(o_names, o_ubs, o_lbs):
                msg += f"            {o_name}: [{o_lb}, {o_ub}]\n"
        msg += f"The state variables that need to be given are as follows:\n"
        e_names = self.var_info.get_external_state_index_for_all().keys()
        for e_name in e_names:
            msg += f"    {e_name}\n"
        msg += "----------------------------------------------------"
        print(msg)

    def get_t(self, idx=-1) -> SpatialArray:
        return self.var_seq[idx, self.var_info.get_index("t [s]")]

    def get_t_seq(self) -> TimeSpatialArray:
        return self.var_seq[:, self.var_info.get_index("t [s]")]

    def get_state(
        self, idx=-1, header=False
    ) -> SpatialArray | tuple[SpatialArray, list[str]]:
        state_idx = self.state_idx["all"]
        state = self.var_seq[idx, state_idx]
        name = [self.var_info.var_names[idx_] for idx_ in state_idx]
        return (state, name) if header else state

    def get_state_seq(
        self, header=False
    ) -> TimeSpatialArray | tuple[TimeSpatialArray, list[str]]:
        state_idx = self.state_idx["all"]
        state = self.var_seq[:, state_idx]
        name = [self.var_info.var_names[idx_] for idx_ in state_idx]
        return (state, name) if header else state

    def get_observation(
        self, idx=-1, header=False
    ) -> SpatialArray | tuple[SpatialArray, list[str]]:
        observation_idx = self.observation_idx["all"]
        observation = self.var_seq[idx, observation_idx]
        name = [self.var_info.var_names[idx_] for idx_ in observation_idx]
        return (observation, name) if header else observation

    def get_observation_seq(
        self, header=False
    ) -> TimeSpatialArray | tuple[TimeSpatialArray, list[str]]:
        observation_idx = self.observation_idx["all"]
        observation = self.var_seq[:, observation_idx]
        name = [self.var_info.var_names[idx_] for idx_ in observation_idx]
        return (observation, name) if header else observation

    def get_state_dim(self) -> int:
        return len(self.var_info.get_state_index_for_all())

    def get_state_upper_bounds(self) -> list[float]:
        return [
            self.var_info.var_upper_bounds[idx]
            for idx in self.var_info.get_state_index_for_all().values()
        ]

    def get_state_lower_bounds(self) -> list[float]:
        return [
            self.var_info.var_lower_bounds[idx]
            for idx in self.var_info.get_state_index_for_all().values()
        ]

    def get_observation_dim(self) -> int:
        return len(self.var_info.get_observation_index_for_all())

    def get_observation_upper_bounds(self) -> list[float]:
        return [
            self.var_info.var_upper_bounds[idx]
            for idx in self.var_info.get_observation_index_for_all().values()
        ]

    def get_observation_lower_bounds(self) -> list[float]:
        return [
            self.var_info.var_lower_bounds[idx]
            for idx in self.var_info.get_observation_index_for_all().values()
        ]

    def get_external_state_dim(self) -> int:
        return len(self.var_info.get_external_state_index_for_all())

    def get_external_state_upper_bounds(self) -> list[float]:
        return [
            self.var_info.var_upper_bounds[idx]
            for idx in self.var_info.get_external_state_index_for_all().values()
        ]

    def get_external_state_lower_bounds(self) -> list[float]:
        return [
            self.var_info.var_lower_bounds[idx]
            for idx in self.var_info.get_external_state_index_for_all().values()
        ]

    def get_log(self) -> pd.DataFrame:
        return self.var_info.get_df()

    def to_csv(self, path: str = "./test.csv") -> None:
        return self.var_info.to_csv(path=path)

    def plot_x0y0(
        self,
        path: str = None,
        scale: float = 1.0,
        h_lim: float = [None, None],
        v_lim: float = [None, None],
        plot_observation: bool = True,
        ext: str = "pdf",
    ):
        # load results
        df = self.get_log()
        df = df[df["terminated"].astype(bool) == False]
        # plot
        fig = plt.figure(figsize=(4.8, 4.8), tight_layout=True)
        ax = fig.add_subplot(111)
        for obj_name in self.ord_obj_names:
            simobj = self.simobjs[obj_name]
            if hasattr(simobj, "axes_x0y0"):
                s_idx = self.state_idx[obj_name]
                o_idx = self.observation_idx[obj_name]
                state_seq = df.to_numpy()[:, s_idx]
                observation_seq = df.to_numpy()[:, o_idx]
                simobj.axes_x0y0(
                    ax,
                    state_seq,
                    observation_seq,
                    scale=scale,
                    plot_observation=plot_observation,
                )
        ax.set_xlabel("$y_{0} \\ \\mathrm{(m)}$")
        ax.set_ylabel("$x_{0} \\ \\mathrm{(m)}$")
        ax.axis("equal")
        if h_lim[0] is not None and h_lim[1] is not None:
            ax.set_xlim(*h_lim)
        if v_lim[0] is not None and v_lim[1] is not None:
            ax.set_ylim(*v_lim)
        if path is None:
            plt.show()
        else:
            os.makedirs(path, exist_ok=True)
            fig.savefig(f"{path}/x0y0.{ext}")

    def plot_timeseries(self, path: str = None, ext: str = "pdf"):
        # load results
        df = self.get_log()
        df = df[df["terminated"].astype(bool) == False]
        for obj_name, simobj in self.simobjs.items():
            if hasattr(simobj, "subplot_timeseries"):
                s_idx = self.state_idx[obj_name]
                o_idx = self.observation_idx[obj_name]
                e_idx = self.external_state_idx[obj_name]
                t_seq = df["t [s]"].to_numpy()
                state_seq = df.to_numpy()[:, s_idx]
                observation_seq = df.to_numpy()[:, o_idx]
                external_state_seq = df.to_numpy()[:, e_idx]
                fig, axes = simobj.subplot_timeseries(
                    t_seq, state_seq, observation_seq, external_state_seq
                )
                if path is None:
                    plt.show()
                else:
                    os.makedirs(path, exist_ok=True)
                    fig.savefig(f"{path}/timeseries_{obj_name}.{ext}")
