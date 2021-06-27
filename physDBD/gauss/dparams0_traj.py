from .dparams0 import DParams0Gauss

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import List, Dict, Tuple

class DParams0GaussTraj:
    
    def __init__(self, times: np.array, dparams0_traj: List[DParams0Gauss]):
        """Time evolution of params

        Args:
            times (np.array): Times (real time) 1D array of length L
            dparams0_traj (List[DParams0Gauss]): List of length L
        """
        self.dparams0_traj = dparams0_traj
        self.times = times

    @property
    def nt(self) -> int:
        """No. timepoints

        Returns:
            int: No. timepoints
        """
        return len(self.dparams0_traj)

    @property
    def nv(self) -> int:
        """No. visible species

        Returns:
            int: No. visible species
        """
        return self.dparams0_traj[0].nv

    def get_tf_outputs(self,
        non_zero_outputs : List[str] = []
        ) -> Dict[str, np.array]:

        outputs = {}
        for i in range(0,len(self.dparams0_traj)):
            output0 = self.dparams0_traj[i].get_tf_output(non_zero_outputs)

            # Put into dict
            for key, val in output0.items():
                if not key in outputs:
                    outputs[key] = []
                outputs[key].append(val)

        # Convert lists to np arrays
        for key, val in outputs.items():
            outputs[key] = np.array(val)

        return outputs
    
    def get_tf_outputs_normalized(self,
        percent: float,
        non_zero_outputs : List[str] = []
        ) -> Tuple[Dict[str, np.array],Dict[str, np.array],Dict[str, np.array]]:

        assert (percent > 0)
        assert (percent <= 1)

        outputs = self.get_tf_outputs(non_zero_outputs)

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

    def convert_to_pd(self) -> pd.DataFrame:
        """Convert to pandas data frame

        Returns:
            pd.DataFrame: DataFrame
        """

        no_params = int(self.nv + self.nv * (self.nv + 1) / 2)
        vals = np.zeros((self.nt, no_params + 1))

        # Time
        data = {}
        data["t"] = self.times

        lf_dicts = [params0.to_lf_dict() for params0 in self.dparams0_traj]
        for key in lf_dicts[0].lf.keys():
            data[key] = [lf[key] for lf in lf_dicts]
        
        return pd.DataFrame.from_dict(data)

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

        dparams0_traj = []
        for t in range(0,no_tpts):

            lf = {key: val[t] for key,val in arr_dict.items()}

            dparams0 = DParams0Gauss.fromLFdict(lf, nv)
            dparams0_traj.append(dparams0)

        return cls(times,dparams0_traj)
