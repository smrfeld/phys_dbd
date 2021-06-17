from .paramsTE import ParamsTE
from .params import Params
from .helpers import convert_np_to_pd

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import List, Dict, Tuple

class ParamsTETraj:
    
    def __init__(self, times: np.array, paramsTE_traj: List[ParamsTE]):
        """Time evolution of params

        Args:
            times (np.array): Times (real time) 1D array of length L
            paramsTE_traj (List[ParamsTE]): List of paramsTE of length L
        """
        self.paramsTE_traj = paramsTE_traj
        self.times = times

    @property
    def nt(self) -> int:
        """No. timepoints

        Returns:
            int: No. timepoints
        """
        return len(self.paramsTE_traj)

    @property
    def nv(self) -> int:
        """No. visible species

        Returns:
            int: No. visible species
        """
        return self.paramsTE_traj[0].nv

    @property
    def nh(self) -> int:
        """No. hidden species

        Returns:
            int: No. hidden species
        """
        return self.paramsTE_traj[0].nh

    def get_tf_outputs_assuming_params0(self,
        non_zero_outputs : List[str] = []
        ) -> Dict[str, np.array]:
        """Get TF outputs assuming they are params0 = std. params. with muh=0,varh_diag=1 and time evolution muhTE=0,varh_diagTE=0

        Args:
            non_zero_outputs (List[str], optional): List of non-zero outputs in long form, i.e. "wt00","wt01", etc.
                If provided only these outputs are returned. Defaults to [].

        Returns:
            Dict[str, np.array]: Keys are long form wt00, wt01, ..., b0, b1, ..., sig2, muh0, muh1, ..., varh_diag0, varh_diag1, .... Values are 1D arrays.
        """
        outputs = {}
        for i in range(0,len(self.paramsTE_traj)):
            output0 = self.paramsTE_traj[i].get_tf_output_assuming_params0(non_zero_outputs)

            # Put into dict
            for key, val in output0.items():
                if not key in outputs:
                    outputs[key] = []
                outputs[key].append(val)

        # Convert lists to np arrays
        for key, val in outputs.items():
            outputs[key] = np.array(val)

        return outputs
    
    def get_tf_outputs_normalized_assuming_params0(self,
        percent: float,
        non_zero_outputs : List[str] = []
        ) -> Tuple[Dict[str, np.array],Dict[str, np.array],Dict[str, np.array]]:
        """Get normalized TF outputs assuming they are params0 = std. params. with muh=0,varh_diag=1 and time evolution muhTE=0,varh_diagTE=0

        Args:
            percent (float): Percentage of data to use to calculate normalization. Should be in (0,1]
            non_zero_outputs (List[str], optional): List of non-zero outputs in long form, i.e. "wt00","wt01", etc.
                If provided only these outputs are returned. Defaults to [].

        Returns:
            Tuple[Dict[str, np.array],Dict[str, np.array],Dict[str, np.array]]: Keys are long form:
                wt00, wt01, ..., b0, b1, ..., sig2, muh0, muh1, ..., varh_diag0, varh_diag1, .... 
                Values are 1D arrays.
                First dict = outputs, second = mean, third = std. dev.
        """

        assert (percent > 0)
        assert (percent <= 1)

        outputs = self.get_tf_outputs_assuming_params0(non_zero_outputs)

        mean = {}
        std_dev = {}

        # Normalization size
        tkey = list(outputs.keys())[0]
        l = len(outputs[tkey])
        norm_size = int(percent * l)
        print("Calculating output normalization from: %d samples" % norm_size)
        idxs = np.arange(0,l)
        idxs_subset = np.random.choice(idxs,size=norm_size,replace=False)

        for key, val in outputs.items():
            val_subset = val[idxs_subset]

            # Mean, std
            mean[key] = np.mean(val_subset,axis=0)
            std_dev[key] = np.std(val_subset,axis=0) + 1e-10

            outputs[key] = (val - mean[key]) / std_dev[key]

        return (outputs, mean, std_dev)

    @classmethod
    def fromArr(cls, times: np.array, arr: np.array, nv: int, nh: int):
        """Construct from array

        Args:
            times (np.array): Times (real time) = 1D array of length T
            arr (np.array): Array of length T x (nv*nh + nv + 1 + 2*muh)
            nv (int): No. visible species
            nh (int): No. hidden species
        """

        # Check length
        if len(times) != len(arr):
            raise ValueError("No timepoints: %d does not match array: %d" % (len(times), len(arr)))

        # Iterate, create
        paramsTE_traj = []
        for arr1d in arr:
            paramsTE = ParamsTE.from1dArr(arr1d, nv, nh)
            paramsTE_traj.append(paramsTE)

        return cls(times, paramsTE_traj)

    @classmethod
    def fromLFdict(cls, times: np.array, lf_dict: Dict[str,np.array], nv: int, nh: int):
        """Construct from long form dictionary

        Args:
            times (np.array): Times (real time) = 1D array of length T
            lf_dict (Dict[str,np.array]): Long form dictionary; keys are:
                wt00, wt01, ..., b0, b1, ..., sig2, muh0, muh1, ..., varh_diag0, varh_diag1, .... 
                Values are 1D arrays of length T
            nv (int): No. visible species
            nh (int): No. hidden species
        """
        # Iterate, create
        paramsTE_traj = []
        for i in range(0,len(times)):
            lf = {}
            for key,vals in lf_dict.items():
                lf[key] = vals[i]
            
            paramsTE = ParamsTE.fromLFdict(lf, nv, nh)
            paramsTE_traj.append(paramsTE)

        return cls(times, paramsTE_traj)

    def convert_to_np(self) -> np.array:
        """Convert to numpy array

        Returns:
            np.array: Array of size T x (nv*nh + nv + 1 + 2*muh) where T is no. timepoints
        """

        # Get length of 1D representation
        l = len(self.paramsTE_traj[0].to_1d_arr())

        # Convert to np array
        arr = np.zeros(shape=(len(self.paramsTE_traj),l))
        for i,params in enumerate(self.paramsTE_traj):
            arr[i] = params.to_1d_arr()

        return arr

    def convert_to_np_with_times(self) -> np.array:
        """Similar to convert_to_np but with additional first column = times (real time)

        Returns:
            np.array: As convert_to_np but with addtional first column = times (real time)
        """
        arr = self.convert_to_np()

        # Add times
        arr = np.transpose(np.concatenate((np.array([self.times]),np.transpose(arr))))

        return arr
    
    def convert_to_pd(self) -> pd.DataFrame:
        """Convert to pandas data frame

        Returns:
            pd.DataFrame: DataFrame
        """

        # To numpy
        arr_with_times = self.convert_to_np_with_times()

        # Convert to pandas
        nv = self.paramsTE_traj[0].nv
        nh = self.paramsTE_traj[0].nh
        return convert_np_to_pd(arr_with_times, nv, nh)

    def export(self, fname: str):
        """Export to human readable CSV file

        Args:
            fname (str): Filename
        """
        
        # Convert to pandas
        df = self.convert_to_pd()

        # Export pandas
        df.to_csv(fname, sep=" ")

    @classmethod
    def fromFile(cls, fname: str, nv: int, nh: int):
        """Import from human readable CSV file

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

            arr1d0 = arr1d[2:]
            params = ParamsTE.from1dArr(arr1d0, nv, nh)
            params_traj.append(params)

        return cls(np.array(times),params_traj)