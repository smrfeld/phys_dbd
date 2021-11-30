from dataclasses import dataclass
import numpy as np

from typing import List

@dataclass
class DataDesc:

    seed_start_inc: int
    seed_end_exc: int

    time_start: float
    time_end: float
    time_interval: float
    times: np.array
    no_times: int

    species: List[str]

    def __init__(self, 
        seed_start_inc: int, 
        seed_end_exc: int,
        time_start: float, 
        time_end: float, 
        time_interval: float, 
        species: List[str]
        ):
        """Data description

        Args:
            seed_start_inc (int): starting seed index inclusive 
            seed_end_exc (int): end seed index exclusive
            time_start (float): time start (real time)
            time_end (float): time end (real time)
            time_interval (float): time interval (real time)
            species (List[str]): list of species
        """
        self.seed_start_inc = seed_start_inc
        self.seed_end_exc = seed_end_exc

        self.time_start = time_start
        self.time_end = time_end
        self.time_interval = time_interval
        self.times = np.arange(self.time_start,self.time_end,self.time_interval)
        self.no_times = len(self.times)

        self.species = species