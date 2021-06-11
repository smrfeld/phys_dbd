from .helpers import convert_np_to_pd, dc_eq
from dataclasses import dataclass
import numpy as np
import pandas as pd

from typing import List

@dataclass(eq=False)
class ParamsTE:

    wt_TE: np.array
    varh_diag_TE: np.array
    b_TE: np.array
    muh_TE: np.array
    sig2_TE: float

    @property
    def nv(self):
        return len(self.b_TE)

    @property
    def nh(self):
        return len(self.muh_TE)

    def __eq__(self, other):
        return dc_eq(self, other)

    def to_1d_arr(self) -> np.array:
        x = np.concatenate([
            self.wt_TE.flatten(),
            self.b_TE,
            np.array([self.sig2_TE]),
            self.muh_TE,
            self.varh_diag_TE
            ])
        return x.flatten()
    
    @classmethod
    def from1dArr(cls, arr: np.array, nv: int, nh: int):
        s = 0
        e = s + nv*nh
        wt_flat = arr[s:e]
        wt = np.reshape(wt_flat,newshape=(nh,nv))

        s = e
        e = s + nv
        b = arr[s:e]

        s = e
        e = s + 1
        sig2 = arr[s:e][0]

        s = e
        e = s + nh
        muh = arr[s:e]

        s = e
        e = s + nh
        varh_diag = arr[s:e]

        return cls(
            wt_TE=wt,
            b_TE=b,
            sig2_TE=sig2,
            muh_TE=muh,
            varh_diag_TE=varh_diag
            )

class ParamsTETraj:
    
    def __init__(self, times: np.array, paramsTE_traj: List[ParamsTE]):
        self.paramsTE_traj = paramsTE_traj
        self.times = times

    @classmethod
    def fromArr(cls, times: np.array, arr: np.array, nv: int, nh: int):

        # Check length
        if len(times) != len(arr):
            raise ValueError("No timepoints: %d does not match array: %d" % (len(times), len(arr)))

        # Iterate, create
        paramsTE_traj = []
        for arr1d in arr:
            paramsTE = ParamsTE.from1dArr(arr1d, nv, nh)
            paramsTE_traj.append(paramsTE)

        return cls(times, paramsTE_traj)

    def convert_to_np(self) -> np.array:

        # Get length of 1D representation
        l = len(self.paramsTE_traj[0].to_1d_arr())

        # Convert to np array
        arr = np.zeros(shape=(len(self.paramsTE_traj),l))
        for i,params in enumerate(self.paramsTE_traj):
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
        nv = self.paramsTE_traj[0].nv
        nh = self.paramsTE_traj[0].nh
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
            params = ParamsTE.from1dArr(arr1d0, nv, nh)
            params_traj.append(params)

        return cls(np.array(times),params_traj)
