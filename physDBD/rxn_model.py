from .net import RxnInputsLayer
from .params import Params
from .params_traj import ParamsTraj

import tensorflow as tf
import numpy as np

from typing import Union, List

import os
import pickle

@tf.keras.utils.register_keras_serializable(package="physDBD")
class RxnModel(tf.keras.Model):

    def __init__(self, 
        nv: int, 
        nh: int, 
        rxn_lyr: RxnInputsLayer, 
        subnet: tf.keras.Model,
        non_zero_outputs : List[str] = []
        ):
        super(RxnModel, self).__init__(name='')

        self.nv = nv
        self.nh = nh
        if len(non_zero_outputs) == 0:
            self.non_zero_outputs = []
            for ih in range(0,nh):
                for iv in range(0,nv):
                    self.non_zero_outputs.append("wt%d%d_TE" % (ih,iv))
            for iv in range(0,nv):
                self.non_zero_outputs.append("b%d_TE" % iv)
            self.non_zero_outputs.append("sig2_TE")
        else:
            self.non_zero_outputs = non_zero_outputs
        
        self.no_outputs = len(self.non_zero_outputs)

        self.rxn_lyr = rxn_lyr
        self.subnet = subnet
        self.output_lyr = tf.keras.layers.Dense(self.no_outputs, activation=None)

        self.rxn_norms_exist = False
        self.rxn_mean = np.array([])
        self.rxn_std_dev = np.array([])

    def get_config(self):
        config = super(RxnModel, self).get_config()

        config.update({
            "nv": self.nv,
            "nh": self.nh,
            "non_zero_outputs": self.non_zero_outputs,
            "no_outputs": self.no_outputs,
            "rxn_lyr": self.rxn_lyr.get_config(),
            "subnet": self.subnet.get_config(),
            "output_ly": self.output_lyr.get_config(),
            "rxn_norms_exist": self.rxn_norms_exist,
            "rxn_mean": self.rxn_mean,
            "rxn_std_dev": self.rxn_std_dev
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __getstate__(self):
        return {
            'nv': self.nv, 
            'nh': self.nh, 
            'non_zero_outputs': self.non_zero_outputs,
            'no_outputs': self.no_outputs,
            'rxn_norms_exist': self.rxn_norms_exist,
            'rxn_mean': self.rxn_mean,
            'rxn_std_dev': self.rxn_std_dev
        }

    '''
    def save(self, dir_name: str):
        # Make directory
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        
        # Dump
        fname = os.path.join(dir_name,"model.txt")
        with open(fname,'wb') as f:
            pickle.dump(self, f)

        # Write subnet
        self.subnet.save(os.path.join(dir_name,"subnet"))

        # Save weights
        self.save_weights(os.path.join(dir_name,"weights.txt"))
    '''

    def integrate(self, 
        params_start: Params, 
        tpt_start: int, 
        no_steps: int,
        time_interval: float
        ) -> ParamsTraj:

        tpts_traj = [tpt_start]
        params_traj = [params_start]
        for _ in range(0,no_steps):
            tpt_curr = tpts_traj[-1]
            input0 = params_start.get_tf_input_assuming_params0(tpt=tpt_curr)
            output0 = self.call(input0)

            # Add to params
            tpt_new = tpt_curr + 1
            params_new = Params.addLFdict(
                params=params_traj[-1],
                lf_dict=output0
                )

            params_traj.append(params_new)
            tpts_traj.append(tpt_new)

        return ParamsTraj(
            times=time_interval*np.array(tpts_traj),
            params_traj=params_traj
            )

    def calculate_rxn_normalization(self, inputs, percent: float):
        # Size
        l = len(inputs["wt"])
        norm_size = int(percent * l)
        print("Calculating input normalization from: %d samples" % norm_size)
        idxs = np.arange(0,l)
        idxs_subset = np.random.choice(idxs,size=norm_size,replace=False)

        inputs_norm = {}
        for key, val in inputs.items():
            inputs_norm[key] = val[idxs_subset]

        x = self.rxn_lyr(inputs_norm)
        self.rxn_mean = np.mean(x,axis=0)
        self.rxn_std_dev = np.std(x,axis=0) + 1e-10
        self.rxn_norms_exist = True

    def call(self, input_tensor, training=False):
        x = self.rxn_lyr(input_tensor)

        if self.rxn_norms_exist:
            x -= self.rxn_mean
            x /= self.rxn_std_dev

        x = self.subnet(x)
        x = self.output_lyr(x)

        # Reshape outputs into dictionary
        out = {}
        for i,non_zero_output in enumerate(self.non_zero_outputs):
            no_tpts = tf.shape(x[:,i])[0]
            out[non_zero_output] = tf.reshape(x[:,i],shape=(no_tpts,1))
        
        return out