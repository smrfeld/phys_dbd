from .params import ParamsGauss
from ..diff_tvr import DiffTVR
from ..helpers import convert_np_to_pd

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import tensorflow as tf

class ParamsGaussTETraj:
    pass

class ParamsGaussTraj:

    def __init__(self, times: np.array, params_traj: List[ParamsGauss]):
        """Trajectory of parameters in time

        Args:
            times (np.array): 1D array of times of length T
            params_traj (List[ParamsGauss]): List of ParamsGauss of length T
        """
        self.params_traj = params_traj
        self.times = times

    @classmethod
    def fromIntegrating(cls, 
        paramsTE_traj: ParamsGaussTETraj, 
        params_init: ParamsGauss, 
        tpt_start: int, 
        no_steps: int,
        constant_vals_lf: Dict[str,float]
        ):

        assert no_steps > 0

        params_traj = [params_init]
        times = [paramsTE_traj.times[tpt_start]]

        for step in range(0,no_steps):
            tpt_curr = tpt_start + step
            tpt_next = tpt_curr + 1

            if tpt_next < len(paramsTE_traj.times):
                times.append(paramsTE_traj.times[tpt_next])
            else:
                times.append(times[-1] + (times[-1] - times[-2]))

            # Add to previous and store
            params = ParamsGauss.addTE(params_traj[-1], paramsTE_traj.paramsTE_traj[tpt_curr])

            params_traj.append(params)
        
        # Constants
        for i in range(0,len(params_traj)):
            for lf,val in constant_vals_lf.items():
                if lf[:2] == "wt":
                    ih = int(lf[2])
                    iv = int(lf[3])
                    params_traj[i].wt[ih,iv] = val
                elif lf[:1] == "b":
                    iv = int(lf[1])
                    params_traj[i].b[iv] = val
                elif lf[:4] == "sig2":
                    params_traj[i].sig2 = val
                elif lf[:3] == "muh":
                    ih = int(lf[3])
                    params_traj[i].muh[ih] = val
                elif lf[:9] == "varh_diag":
                    ih = int(lf[9])
                    params_traj[i].varh_diag[ih] = val
                else:
                    raise ValueError("LF: %s not recognized!" % lf)

        return cls(
            times=times,
            params_traj=params_traj
            )

    @classmethod
    def fromDataStd(cls, data: np.array, times: np.array, nh: int):
        params_traj = []
        for data0 in data:
            params = ParamsGauss.fromDataStd(data0, nh)
            params_traj.append(params)
        return cls(times, params_traj)

    @property
    def nt(self) -> int:
        """No. timepoints

        Returns:
            int: No. timepoints
        """
        return len(self.params_traj)

    @property
    def n(self) -> int:
        """No. species

        Returns:
            int: No. species
        """
        return self.params_traj[0].n
    
    @property
    def nv(self) -> int:
        """No. visible species

        Returns:
            int: No. visible species
        """
        return self.params_traj[0].nv

    @property
    def nh(self) -> int:
        """No. hidden species

        Returns:
            int: No. hidden species
        """
        return self.params_traj[0].nh

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
    
    def convert_to_pd(self) -> pd.DataFrame:
        """Convert to pandas data frame

        Returns:
            pd.DataFrame: The pandas data frame
        """

        no_params = int(self.n + self.n * (self.n + 1) / 2)
        vals = np.zeros((self.nt, no_params + 1))

        idx = 0
        columns = []

        # Time
        columns.append("t")
        for t in range(0,self.nt):
            vals[t,idx] = self.times[t]
        idx += 1

        # Mean
        for i in range(0,self.n):
            s = "mu_%d" % i
            columns.append(s)

            for t in range(0,self.nt):
                vals[t,idx] = self.params_traj[t].mu[i]
            
            idx += 1
        
        # Chol
        for i in range(0,self.n):
            for j in range(0,i+1):
                s = "chol_%d_%d" % (i,j)
                columns.append(s)

                for t in range(0,self.nt):
                    vals[t,idx] = self.params_traj[t].chol[i,j]

                idx += 1

        return pd.DataFrame(vals, columns=columns)

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
    def fromFile(cls, fname: str, nv: int, nh: int):
        """Import from human-readable CSV file. First row should be header

        Args:
            fname (str): Filename
            nv (int): No. visible species
            nh (int): No. hidden species
        """

        # Import
        df = pd.read_csv(fname, sep=" ")

        # To numpy
        arr = df.to_numpy()

        params_traj = []
        times = []
        for arr1d in arr:
            t = arr1d[1]
            times.append(t)

            params = ParamsGauss.from1dArr(
                nv=nv,
                nh=nh,
                columns=df.columns[2:],
                arr=arr1d[2:]
                )
            params_traj.append(params)

        return cls(np.array(times),params_traj)

    def differentiate_with_TVR(self, 
        alphas: Dict[str,float], 
        no_opt_steps: int, 
        non_zero_vals: List[str] = []
        ) -> ParamsGaussTETraj:

        n = len(self.params_traj)

        diff_tvr = DiffTVR(n=n,dx=1.0)

        lf_dict = self.convert_to_lf_dict()

        deriv_guess = np.zeros(n-1)

        lf_derivs = {}
        
        for lf, arr in lf_dict.items():
            if len(non_zero_vals) == 0 or lf in non_zero_vals:
                
                lf_derivs[lf] = diff_tvr.get_deriv_tvr(
                    data=arr,
                    deriv_guess=deriv_guess,
                    alpha=alphas[lf],
                    no_opt_steps=no_opt_steps,
                    return_progress=False
                    )[0]

            else:

                # Zero
                lf_derivs[lf] = np.zeros(n-1)

        return ParamsGaussTETraj.fromLFdict(self.times[:-1], lf_derivs, self.nv, self.nh)