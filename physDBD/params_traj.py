from .paramsTE_traj import ParamsTETraj
from .params import Params
from .diff_tvr import DiffTVR
from .helpers import convert_np_to_pd

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import tensorflow as tf

class ParamsTraj:

    def __init__(self, times: np.array, params_traj: List[Params]):
        self.params_traj = params_traj
        self.times = times

    def get_tf_inputs_assuming_params0(self) -> Dict[str, np.array]:
        inputs = {}
        for tpt in range(0,len(self.params_traj)):

            # Get input
            input0 = self.params_traj[tpt].get_tf_input_assuming_params0(tpt)
            
            # Put into dict
            for key, val in input0.items():
                if not key in inputs:
                    inputs[key] = []
                inputs[key].append(val[0])

        # Convert lists to np arrays
        for key, val in inputs.items():
            inputs[key] = np.array(val)

        return inputs

    @classmethod
    def fromPCA(cls, data: np.array, times: np.array, muh: np.array, varh_diag: np.array):
        params_traj = []
        for data0 in data:
            params = Params.fromPCA(data0, muh, varh_diag)
            params_traj.append(params)
        return cls(times, params_traj)

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

    def convert_to_lf_dict(self) -> Dict[str,np.array]:
        lf_dict = {}
        for params in self.params_traj:
            lf = params.to_lf_dict()
            for key,val in lf.items():
                if not key in lf_dict:
                    lf_dict[key] = []
                lf_dict[key].append(val)

        for key in lf_dict.keys():
            lf_dict[key] = np.array(lf_dict[key])

        return lf_dict

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

    def differentiate_with_TVR(self, alphas: Dict[str,float], no_opt_steps: int, latents_are_std: bool = True) -> ParamsTETraj:
        n = len(self.params_traj)

        diff_tvr = DiffTVR(n=n,dx=1.0)

        lf_dict = self.convert_to_lf_dict()

        deriv_guess = np.zeros(n+1)

        lf_derivs = {}
        for lf, arr in lf_dict.items():
            if latents_are_std:
                if len(lf) >= 3 and (lf[:3] == "muh" or lf[:4] == "varh"):
                    lf_derivs[lf] = np.zeros(n+1)
                    continue

            lf_derivs[lf] = diff_tvr.get_deriv_tvr(
                data=arr,
                deriv_guess=deriv_guess,
                alpha=alphas[lf],
                no_opt_steps=no_opt_steps,
                return_progress=False
                )[0]

            # Drop last (trapezoidal rule => bad length)
            lf_derivs[lf] = lf_derivs[lf][:-1]

        return ParamsTETraj.fromLFdict(self.times, lf_derivs, self.nv, self.nh)