import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d

from . import Simulator
from .obj_ship_actuator import *
from .obj_ship_mmg import *
from .obj_obstacle import *
from .obj_wind import *


class PresetSimulator(Simulator):

    def __init__(
        self,
        dt_act=1,
        dt_sim=0.1,
        collide_checker_type=None,
        ship_type="MMGEssoOsaka3m",
        obstacle_type="InukaiPond",
        wind_type="RandomWind",
        ship_args={"solve_method": "euler", "interp_method": "linear", "f2py": False},
        verbose=True,
    ):
        simobjs = []
        simobjs += self.get_ship_obj(ship_type, ship_args)
        simobjs += self.get_obstacle_obj(obstacle_type)
        simobjs += self.get_wind_obj(wind_type)
        super().__init__(
            *simobjs,
            dt_act=dt_act,
            dt_sim=dt_sim,
            collide_checker_type=collide_checker_type,
        )
        if verbose:
            self.get_info()

    def get_ship_obj(self, ship_type, ship_args):
        simobjs = []
        if ship_type == "MMGEssoOsaka3m":
            simobjs.append(MMGEssoOsaka3m(**ship_args))
        elif ship_type == "MMGACTEssoOsaka3m":
            simobjs.append(MMGEssoOsaka3m(**ship_args))
            simobjs.append(Rudder())
            simobjs.append(Propeller())
        elif ship_type == "MMGTakaoki3m":
            simobjs.append(MMGTakaoki3m(**ship_args))
        elif ship_type == "MMGACTTakaoki3m":
            simobjs.append(MMGTakaoki3m(**ship_args))
            simobjs.append(VecTwinRudder())
            simobjs.append(Propeller())
            simobjs.append(BowThruster())
        else:
            raise ValueError(f"Unknown ship type: {ship_type}")
        return simobjs

    def get_obstacle_obj(self, obstacle_type):
        simobjs = []
        if obstacle_type == "InukaiPond":
            simobjs.append(InukaiPond())
        elif obstacle_type == "StraightBerth":
            simobjs.append(StraightBerth())
        elif obstacle_type == "CornerBerth":
            simobjs.append(CornerBerth())
        elif obstacle_type == "OpenSea":
            pass
        else:
            raise ValueError(f"Unknown obstacle type: {obstacle_type}")
        return simobjs

    def get_wind_obj(self, wind_type):
        simobjs = []
        if wind_type == "RandomWind":
            simobjs.append(RandomWind())
        elif wind_type == "StationaryWind":
            simobjs.append(StationaryWind())
        else:
            raise ValueError(f"Unknown wind type: {wind_type}")
        return simobjs

    def steps(self, t_seq: npt.ArrayLike, action_seq: npt.ArrayLike) -> pd.DataFrame:
        # preparation
        action_seq_interp = interp1d(
            np.array(t_seq),
            np.array(action_seq).T,
            kind="previous",
            fill_value="extrapolate",
        )
        N_step = int((t_seq[-1] - self.get_t()) / self.dt_act) + 1
        # start simulation
        for _ in range(N_step):
            action = action_seq_interp(self.get_t())
            _, terminated, _ = self.step(action)
            if terminated:
                break
        return self.get_log()
