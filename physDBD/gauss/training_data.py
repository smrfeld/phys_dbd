from .dparams0_traj import DParams0GaussTraj
from .params0_traj import Params0GaussTraj
import numpy as np

from typing import Dict, List, Tuple

from dataclasses import dataclass

import pickle

from enum import Enum

import os

def join_dicts(existing_dict, new_dict):
    for key, val in new_dict.items():
        if not key in existing_dict:
            existing_dict[key] = val
        else:
            existing_dict[key] = np.concatenate((existing_dict[key],val))
    return existing_dict

class DataType(Enum):
    TRAINING = 0
    VALIDATION = 1

@dataclass(eq=False)
class TrainingGaussData:

    train_inputs : Dict[str,np.array]
    valid_inputs : Dict[str,np.array]

    train_outputs_not_stdrd : Dict[str,np.array]
    valid_outputs_not_stdrd : Dict[str,np.array]

    train_outputs_mean : Dict[str,float]
    train_outputs_std_dev : Dict[str,float]

    train_outputs_stdrd : Dict[str,np.array]
    valid_outputs_stdrd : Dict[str,np.array]

    def __init__(self):
        """Constructor
        """
        self.train_inputs = {}
        self.valid_inputs = {}

        self.train_outputs_not_stdrd = {}
        self.valid_outputs_not_stdrd = {}

        self.train_outputs_mean = {}
        self.train_outputs_std_dev = {}

        self.train_outputs_stdrd = {}
        self.valid_outputs_stdrd = {}

    def reap_params0_traj_for_inputs(self, 
        params0_traj: Params0GaussTraj, 
        data_type: DataType,
        non_zero_idx_pairs_vv: List[Tuple[int,int]]
        ):
        """Reap inputs from a ParamTraj at all timepoints by calling get_tf_inputs_assuming_params0
            Fills out train_inputs, valid_inputs

        Args:
            params0_traj (Params0GaussTraj): The param trajectory
            data_type (DataType): Training vs validation
            non_zero_idx_pairs_vv (List[Tuple[int,int]]): Non-zero idx pairs in visible part of precision matrix
        """
        inputs0 = params0_traj.get_tf_inputs(non_zero_idx_pairs_vv)
        if data_type == DataType.TRAINING:
            self.train_inputs = join_dicts(self.train_inputs, inputs0)
        elif data_type == DataType.VALIDATION:
            self.valid_inputs = join_dicts(self.valid_inputs, inputs0)

    def reap_dparams0_traj_for_ouputs(self, 
        dparams0_traj: DParams0GaussTraj, 
        data_type: DataType, 
        non_zero_outputs: List[str] = []
        ):
        """Reap outputs from a DParams0GaussTraj at all timepoints by calling get_tf_outputs_assuming_params0

        Args:
            dparams0_traj (DParams0GaussTraj): The derivatives trajectory
            data_type (DataType): Training vs validation
            non_zero_outputs (List[str]): See DParams0GaussTraj.get_tf_outputs. Defaults to [].
        """
        outputs0 = dparams0_traj.get_tf_outputs(non_zero_outputs=non_zero_outputs)
        print(outputs0)
        if data_type == DataType.TRAINING:
            self.train_outputs_not_stdrd = join_dicts(self.train_outputs_not_stdrd, outputs0)
        elif data_type == DataType.VALIDATION:
            self.valid_outputs_not_stdrd = join_dicts(self.valid_outputs_not_stdrd, outputs0)

    def calculate_output_standardizations_and_apply(self, percent: float):
        """Calculate output standardizations and apply them
            Constructs train_outputs_stdrd, valid_outputs_stdrd from 
            train_outputs_not_stdrd, valid_outputs_not_stdrd

        Args:
            percent (float): Percent of data to calculate mean, std. dev from
        """

        # Normalization size
        norm_size = int(percent * len(self.train_outputs_not_stdrd["dmu_v_0"]))
        print("Calculating output normalization from: %d samples" % norm_size)

        # Normalize training
        self.train_outputs_mean = {}
        self.train_outputs_std_dev = {}
        for key, val in self.train_outputs_not_stdrd.items():
            idxs = np.arange(0,len(val))
            idxs_subset = np.random.choice(idxs,size=norm_size,replace=False)
            val_subset = val[idxs_subset]

            # Mean, std
            self.train_outputs_mean[key] = np.mean(val_subset,axis=0)
            self.train_outputs_std_dev[key] = np.std(val_subset,axis=0)

            if abs(self.train_outputs_mean[key]) < 1e-6:
                self.train_outputs_mean[key] = 0.0
            if abs(self.train_outputs_std_dev[key]) < 1e-6:
                self.train_outputs_std_dev[key] = 1.0

        # Apply stdardization
        self._standardize_ouputs(self.train_outputs_mean, self.train_outputs_std_dev)

    def _standardize_ouputs(self, mean: Dict[str,np.array], std_dev: Dict[str,np.array]):
        """Internal helper to construct train_outputs_stdrd, valid_outputs_stdrd
            from train_outputs_not_stdrd, valid_outputs_not_stdrd

        Args:
            mean (Dict[str,np.array]): Mean
            std_dev (Dict[str,np.array]): Std. dev.
        """
        self.train_outputs_stdrd = {}
        self.valid_outputs_stdrd = {}
        for key, val in self.train_outputs_not_stdrd.items():
            self.train_outputs_stdrd[key] = (val - mean[key]) / std_dev[key]
        for key, val in self.valid_outputs_not_stdrd.items():
            self.valid_outputs_stdrd[key] = (val - mean[key]) / std_dev[key]

    def write_output_standardizations(self, dir_name: str):
        """Write standardizations mean and std. dev. with pickle.dump

        Args:
            dir_name (str): Directory name, files are "cache_outputs_mean.txt" 
                and "cache_outputs_std_dev.txt"
        """
        # Save the output mean/std dev
        fname = os.path.join(dir_name, "cache_outputs_mean.txt")
        with open(fname,'wb') as f:
            pickle.dump(self.train_outputs_mean, f)

        fname = os.path.join(dir_name, "cache_outputs_std_dev.txt")
        with open(fname,'wb') as f:
            pickle.dump(self.train_outputs_std_dev, f)

    def read_output_standardizations(self, dir_name: str):
        """Read standardizations back with pickle.load
            Does NOT calculate train_outputs_stdrd, valid_outputs_stdrd automatically
            Use read_output_standardizations_and_apply for that

        Args:
            dir_name (str): Directory name, files are "cache_outputs_mean.txt" 
                and "cache_outputs_std_dev.txt"
        """
        # Try to load
        fname = os.path.join(dir_name, "cache_outputs_mean.txt")
        with open(fname,'rb') as f:
            self.train_outputs_mean = pickle.load(f)
        
        fname = os.path.join(dir_name, "cache_outputs_std_dev.txt")
        with open(fname,'rb') as f:
            self.train_outputs_std_dev = pickle.load(f)

    def read_output_standardizations_and_apply(self, dir_name: str):
        """Read standardizations back with pickle.load
            AND automatically calculates train_outputs_stdrd, valid_outputs_stdrd

        Args:
            dir_name (str): Directory name, files are "cache_outputs_mean.txt" 
                and "cache_outputs_std_dev.txt"
        """
        self.read_output_standardizations(dir_name)

        # Apply standardization
        self._standardize_ouputs(self.train_outputs_mean, self.train_outputs_std_dev)

