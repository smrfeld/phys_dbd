from dataclasses import dataclass
import numpy as np

@dataclass
class DataDesc:

    no_seeds: int
    vol_exp: float
    tpt_start: int
    tpt_max: int
    tpt_interval: int
    dt: float
    tpts: np.array
    times: np.array
    no_tpts: int

    def __init__(self, 
        no_seeds: int, 
        vol_exp: float, 
        tpt_start: int, 
        tpt_max: int, 
        tpt_interval: int, 
        dt: float
        ):
        self.no_seeds = no_seeds
        self.vol_exp = vol_exp
        self.tpt_start = tpt_start
        self.tpt_max = tpt_max
        self.tpt_interval = tpt_interval
        self.dt = dt

        self.tpts = np.arange(tpt_start, tpt_max, tpt_interval)
        self.times = self.dt * self.tpts
        self.no_tpts = len(self.times)