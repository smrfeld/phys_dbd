from physDBD.paramsTE_traj import ParamsTETraj
from physDBD.paramsTE import ParamsTE
from .params_traj import ParamsTraj
import tensorflow as tf
import numpy as np

from typing import Dict, List

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
class TrainingData:

    train_inputs : Dict[str,np.array]
    valid_inputs : Dict[str,np.array]

    train_outputs_not_stdrd : Dict[str,np.array]
    valid_outputs_not_stdrd : Dict[str,np.array]

    train_outputs_mean : Dict[str,float]
    train_outputs_std_dev : Dict[str,float]

    train_outputs_stdrd : Dict[str,np.array]
    valid_outputs_stdrd : Dict[str,np.array]

    def __init__(self):
        self.train_inputs = {}
        self.valid_inputs = {}

        self.train_outputs_not_stdrd = {}
        self.valid_outputs_not_stdrd = {}

        self.train_outputs_mean = {}
        self.train_outputs_std_dev = {}

        self.train_outputs_stdrd = {}
        self.valid_outputs_stdrd = {}

    def reap_params_traj_for_inputs(self, params_traj: ParamsTraj, data_type: DataType):
        inputs0 = params_traj.get_tf_inputs_assuming_params0()
        if data_type == DataType.TRAINING:
            self.train_inputs = join_dicts(self.train_inputs, inputs0)
        elif data_type == DataType.VALIDATION:
            self.valid_inputs = join_dicts(self.valid_inputs, inputs0)

    def reap_paramsTE_traj_for_ouputs(self, paramsTE_traj: ParamsTETraj, data_type: DataType, non_zero_outputs: List[str]):
        outputs0 = paramsTE_traj.get_tf_outputs_assuming_params0(non_zero_outputs=non_zero_outputs)
        if data_type == DataType.TRAINING:
            self.train_outputs_not_stdrd = join_dicts(self.train_outputs_not_stdrd, outputs0)
        elif data_type == DataType.VALIDATION:
            self.valid_outputs_not_stdrd = join_dicts(self.valid_outputs_not_stdrd, outputs0)

    def calculate_output_standardizations(self, percent: float):

        # Normalization size
        norm_size = int(percent * len(self.train_outputs_not_stdrd["wt00_TE"]))
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
        self.train_outputs_stdrd = {}
        self.valid_outputs_stdrd = {}
        for key, val in self.train_outputs_not_stdrd.items():
            self.train_outputs_stdrd[key] = (val - mean[key]) / std_dev[key]
        for key, val in self.valid_outputs_not_stdrd.items():
            self.valid_outputs_stdrd[key] = (val - mean[key]) / std_dev[key]

    def write_output_standardizations(self, dir_name: str):
        # Save the output mean/std dev
        fname = os.path.join(dir_name, "cache_outputs_mean.txt")
        with open(fname,'wb') as f:
            pickle.dump(self.train_outputs_mean, f)

        fname = os.path.join(dir_name, "cache_outputs_std_dev.txt")
        with open(fname,'wb') as f:
            pickle.dump(self.train_outputs_std_dev, f)

    def read_output_standardizations(self, dir_name: str):
        # Try to load
        fname = os.path.join(dir_name, "cache_outputs_mean.txt")
        with open(fname,'rb') as f:
            self.train_outputs_mean = pickle.load(f)
        
        fname = os.path.join(dir_name, "cache_outputs_std_dev.txt")
        with open(fname,'rb') as f:
            self.train_outputs_std_dev = pickle.load(f)

    def read_output_standardizations_and_apply(self, dir_name: str):
        self.read_output_standardizations(dir_name)

        # Apply standardization
        self._standardize_ouputs(self.train_outputs_mean, self.train_outputs_std_dev)

