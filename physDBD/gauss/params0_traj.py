from .dparams0_traj import DParams0GaussTraj
from .params0 import Params0Gauss
from ..diff_tvr import DiffTVR

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import tensorflow as tf

class Params0GaussTraj:

    def __init__(self, times: np.array, params0_traj: List[Params0Gauss]):
        """Trajectory of parameters in time

        Args:
            times (np.array): 1D array of times of length T
            params0_traj (List[Params0Gauss]): List of Params0Gauss of length T
        """
        self.params0_traj = params0_traj
        self.times = times

    @classmethod
    def fromIntegrating(cls, 
        dparams0_traj: DParams0GaussTraj, 
        params0_init: Params0Gauss, 
        tpt_start: int, 
        no_steps: int,
        constant_vals_lf: Dict[str,float]
        ):

        assert no_steps > 0

        params0_traj = [params0_init]
        times = [dparams0_traj.times[tpt_start]]

        for step in range(0,no_steps):
            tpt_curr = tpt_start + step
            tpt_next = tpt_curr + 1

            if tpt_next < len(dparams0_traj.times):
                times.append(dparams0_traj.times[tpt_next])
            else:
                times.append(times[-1] + (times[-1] - times[-2]))

            # Add to previous and store
            params = Params0Gauss.addTE(params0_traj[-1], dparams0_traj.dparams0TE_traj[tpt_curr])

            params0_traj.append(params)
        
        # Constants
        for i in range(0,len(params0_traj)):
            for lf,val in constant_vals_lf.items():
                s = lf.split('_')
                if s[0] == "mu" and s[1] == "v" and s[2].isdigit():
                    iv = int(s[2])
                    params0_traj[i].mu_v[iv] = val
                elif s[0] == "chol" and s[1] == "v" and s[2].isdigit() and s[3].isdigit():
                    iv = int(s[2])
                    jv = int(s[3])
                    params0_traj[i].chol_v[iv,jv] = val
                else:
                    raise ValueError("LF: %s not recognized!" % lf)

        return cls(
            times=times,
            params0_traj=params0_traj
            )
            
    def get_tf_inputs(self, non_zero_idxs_vv: List[Tuple[int,int]]) -> Dict[str, np.array]:
        inputs = {}
        for tpt in range(0,len(self.params0_traj)-1): # Take off one to match derivatives

            # Get input
            input0 = self.params0_traj[tpt].get_tf_input(
                tpt=tpt, 
                non_zero_idxs_vv=non_zero_idxs_vv
                )
            
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
    def fromData(cls, data: np.array, times: np.array):
        """Construct from applying PCA to data array

        Args:
            data (np.array): Data array of size (no_times, no_seeds, no_species)
            times (np.array): 1D array of times (real-time)
        """
        params0_traj = []
        for data0 in data:
            params = Params0Gauss.fromData(data0)
            params0_traj.append(params)
        return cls(times, params0_traj)

    @property
    def nt(self) -> int:
        """No. timepoints

        Returns:
            int: No. timepoints
        """
        return len(self.params0_traj)
    
    @property
    def nv(self) -> int:
        """No. visible species

        Returns:
            int: No. visible species
        """
        return self.params0_traj[0].nv
    
    def convert_to_pd(self) -> pd.DataFrame:
        """Convert to pandas data frame

        Returns:
            pd.DataFrame: The pandas data frame
        """

        no_params = int(self.nv + self.nv * (self.nv + 1) / 2)
        vals = np.zeros((self.nt, no_params + 1))

        # Time
        data = {}
        data["t"] = self.times

        lf_dicts = [params0.to_lf_dict() for params0 in self.params0_traj]
        for key in lf_dicts[0].keys():
            data[key] = [lf[key] for lf in lf_dicts]
        
        return pd.DataFrame.from_dict(data)

    def export(self, fname: str):
        """Export to human-readable CSV file

        Args:
            fname (str): Filename
        """
        
        # Convert to pandas
        df = self.convert_to_pd()

        # Export pandas
        df.to_csv(fname, sep=" ")

    @classmethod
    def fromFile(cls, fname: str, nv: int):
        """Import from human-readable CSV file. First row should be header

        Args:
            fname (str): Filename
            nv (int): No. visible species
        """

        # Import
        df = pd.read_csv(fname, sep=" ")

        # To dict of np arrays
        arr_dict = df.to_dict('list')
        del arr_dict["Unnamed: 0"]

        times = arr_dict["t"]
        del arr_dict["t"]
        no_tpts = len(times)

        params0_traj = []
        for t in range(0,no_tpts):

            lf = {key: val[t] for key,val in arr_dict.items()}

            params0 = Params0Gauss.fromLFdict(lf=lf, nv=nv)
            params0_traj.append(params0)

        return cls(times,params0_traj)

    def differentiate_with_TVR(self, 
        alphas: Dict[str,float], 
        no_opt_steps: int, 
        non_zero_vals: List[str] = []
        ) -> DParams0GaussTraj:
        n = len(self.params0_traj)

        diff_tvr = DiffTVR(n=n,dx=1.0)

        params0LF_traj = [params0.to_lf_dict() for params0 in self.params0_traj]

        deriv_guess = np.zeros(n-1)

        lf_derivs = {}
        
        for key in params0LF_traj[0].keys():
            if len(non_zero_vals) == 0 or key in non_zero_vals:
                
                arr = np.array([p0[key] for p0 in params0LF_traj])

                lf_derivs["d"+key] = diff_tvr.get_deriv_tvr(
                    data=arr,
                    deriv_guess=deriv_guess,
                    alpha=alphas[key],
                    no_opt_steps=no_opt_steps,
                    return_progress=False
                    )[0]

            else:

                # Zero
                lf_derivs["d"+key] = np.zeros(n-1)

        return DParams0GaussTraj.fromLFdict(
            times=self.times[:-1], 
            lf_dict=lf_derivs, 
            nv=self.nv
            )