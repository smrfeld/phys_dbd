from .paramsTE import ParamsTE
from .helpers import convert_np_to_pd

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import List, Dict

class ParamsTETraj:
    
    def __init__(self, times: np.array, paramsTE_traj: List[ParamsTE]):
        self.paramsTE_traj = paramsTE_traj
        self.times = times

    def get_tf_outputs_assuming_params0(self) -> Dict[str, np.array]:
        outputs = {}
        for i in range(0,len(self.paramsTE_traj)):
            output0 = self.paramsTE_traj[i].get_tf_output_assuming_params0()

            # Put into dict
            for key, val in output0.items():
                if not key in outputs:
                    outputs[key] = []
                outputs[key].append(val)

        # Convert lists to np arrays
        for key, val in outputs.items():
            outputs[key] = np.array(val)

        return outputs
    
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
