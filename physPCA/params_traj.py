from .paramsTE import ParamsTETraj
from .params import Params
from .diff_tvr import DiffTVR
from .helpers import convert_np_to_pd

import pandas as pd
import numpy as np
from typing import List, Tuple

class ParamsTraj:

    def __init__(self, times: np.array, params_traj: List[Params]):
        self.params_traj = params_traj
        self.times = times

    @property
    def nv(self):
        return self.params_traj[0].nv

    @property
    def nh(self):
        return self.params_traj[0].nh

    def convert_to_np(self) -> np.array:

        # Get length of 1D representation
        l = len(self.params_traj[0].to_1d_arr())

        # Convert to np array
        arr = np.zeros(shape=(len(self.params_traj),l))
        for i,params in enumerate(self.params_traj):
            arr[i] = params.to_1d_arr()

        return arr

    def convert_to_np_with_times(self) -> np.array:
        arr = self.convert_to_np()

        # Add times
        arr = np.transpose(np.concatenate((np.array([self.times]),np.transpose(arr))))

        return arr

    def convert_to_pd(self) -> pd.DataFrame:

        # To numpy
        arr_with_times = self.convert_to_np_with_times()

        # Convert to pandas
        nv = self.params_traj[0].nv
        nh = self.params_traj[0].nh
        return convert_np_to_pd(arr_with_times, nv, nh)
        
    def export(self, fname: str):
        
        # Convert to pandas
        df = self.convert_to_pd()

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

        return cls(np.array(times),params_traj)

    def differentiate_with_TVR(self, alpha: float, no_opt_steps: int) -> ParamsTETraj:
        n = len(self.params_traj)

        diff_tvr = DiffTVR(n=n,dx=1.0)

        arr = self.convert_to_np()

        deriv_guess = np.zeros(n+1)

        d = arr.shape[1]
        deriv_arr = np.zeros(shape=(n+1,d))
        for i in range(0,d):
            deriv_arr[:,i] = diff_tvr.get_deriv_tvr(
                data=arr[:,i],
                deriv_guess=deriv_guess,
                alpha=alpha,
                no_opt_steps=no_opt_steps,
                return_progress=False
                )[0]

        # Drop last (trapezoidal rule => bad length)
        deriv_arr = deriv_arr[:-1]

        return ParamsTETraj.fromArr(self.times,deriv_arr,self.nv,self.nh)