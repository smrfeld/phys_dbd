from dataclasses import dataclass
import numpy as np

from typing import List

@dataclass
class DataDesc:

    no_seeds: int

    time_start: float
    time_end: float
    time_interval: float
    times: np.array
    no_times: int

    species: List[str]

    def __init__(self, 
        no_seeds: int, 
        time_start: float, 
        time_end: float, 
        time_interval: float, 
        species: List[str]
        ):
        """Data description

        Args:
            no_seeds (int): no. seeds
            time_start (float): time start (real time)
            time_end (float): time end (real time)
            time_interval (float): time interval (real time)
            species (List[str]): list of species
        """
        self.no_seeds = no_seeds

        self.time_start = time_start
        self.time_end = time_end
        self.time_interval = time_interval
        self.times = np.arange(self.time_start,self.time_end,self.time_interval)
        self.no_times = len(self.times)

        self.species = species