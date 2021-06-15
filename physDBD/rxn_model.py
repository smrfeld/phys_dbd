from .net import RxnInputsLayer

import tensorflow as tf
import numpy as np

from typing import Union, List

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
            out[non_zero_output] = tf.reshape(x[:,i],shape=(len(x[:,i]),1))
        
        return out