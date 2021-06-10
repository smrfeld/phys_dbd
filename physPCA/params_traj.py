from .params import Params

import pandas as pd
import numpy as np
from typing import List, Tuple

class ParamsTraj:

    def __init__(self, times: np.array, params_traj: List[Params]):
        self.params_traj = params_traj
        self.times = times

    def convert_params_traj_to_pd(self) -> pd.DataFrame:

        # Get length of 1D representation
        l = len(self.params_traj[0].to_1d_arr())

        # Convert to np array
        arr = np.zeros(shape=(len(self.params_traj),l))
        for i,params in enumerate(self.params_traj):
            arr[i] = params.to_1d_arr()

        # Add times
        arr = np.transpose(np.concatenate((np.array([self.times]),np.transpose(arr))))

        # Convert to pandas
        nv = self.params_traj[0].nv
        nh = self.params_traj[0].nh
        columns = ["t"]
        for ih in range(0,nh):
            for iv in range(0,nv):
                columns += ["wt%d%d" % (ih,iv)]
        for iv in range(0,nv):
            columns += ["b%d" % iv]
        columns += ["sig2"]
        for ih in range(0,nh):
            columns += ["muh%d" % ih]
        for ih in range(0,nh):
            columns += ["varh_diag%d" % ih]

        df = pd.DataFrame(arr, columns=columns)
        return df

    def export_params_traj(self, fname: str):
        
        # Convert to pandas
        df = self.convert_params_traj_to_pd()

        # Export pandas
        df.to_csv(fname, sep=" ")

    @classmethod
    def fromFile(cls, fname: str, nv: int, nh: int):

        # Import
        df = pd.read_csv(fname, sep=" ")

        # To numpy
        arr = df.to_numpy()

        params_traj = []
        times = []
        for arr1d in arr:
            t = arr1d[1]
            times.append(t)

            arr1d0 = arr1d[2:]
            params = Params.from1dArr(arr1d0, nv, nh)
            params_traj.append(params)

        return ParamsTraj(np.array(times),params_traj)