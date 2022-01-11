from .net_fully_visible import RxnInputsGaussLayerFV

import tensorflow as tf
import numpy as np

from typing import List

@tf.keras.utils.register_keras_serializable(package="physDBD")
class RxnGaussModelFV(tf.keras.Model):

    @classmethod
    def construct(cls, 
        nv: int, 
        rxn_lyr: RxnInputsGaussLayerFV, 
        subnet: tf.keras.Model,
        non_zero_outputs : List[str] = [],
        rxn_norms_exist : bool = False,
        rxn_mean : np.array = np.array([]),
        rxn_std_dev : np.array = np.array([]),
        **kwargs
        ):

        if len(non_zero_outputs) == 0:
            non_zero_outputs_use = []
            for i in range(0,nv):
                for j in range(0,i+1):
                    s = "dchol_v_%d_%d" % (i,j)
                    non_zero_outputs_use.append(s)
            for i in range(0,nv):
                s = "dmu_v_%d" % i
                non_zero_outputs_use.append(s)
        else:
            non_zero_outputs_use = non_zero_outputs
        
        no_outputs = len(non_zero_outputs_use)

        output_lyr = tf.keras.layers.Dense(no_outputs, activation=None)

        return cls(
            nv=nv,
            rxn_lyr=rxn_lyr,
            subnet=subnet,
            output_lyr=output_lyr,
            non_zero_outputs=non_zero_outputs_use,
            rxn_norms_exist=rxn_norms_exist,
            rxn_mean=rxn_mean,
            rxn_std_dev=rxn_std_dev,
            **kwargs
            )

    def __init__(self, 
        nv: int, 
        rxn_lyr: RxnInputsGaussLayerFV, 
        subnet: tf.keras.Model,
        output_lyr: tf.keras.layers.Layer,
        non_zero_outputs : List[str],
        rxn_norms_exist : bool = False,
        rxn_mean : np.array = np.array([]),
        rxn_std_dev : np.array = np.array([]),
        **kwargs
        ):
        super(RxnGaussModelFV, self).__init__(**kwargs)

        self.nv = nv

        self.non_zero_outputs = non_zero_outputs
        self.no_outputs = len(self.non_zero_outputs)

        self.rxn_lyr = rxn_lyr
        self.subnet = subnet
        self.output_lyr = output_lyr

        self.rxn_norms_exist = rxn_norms_exist
        self.rxn_mean = rxn_mean
        self.rxn_std_dev = rxn_std_dev

    def get_config(self):
        # Do not call super.get_config !!!
        config = {
            "nv": self.nv,
            "non_zero_outputs": self.non_zero_outputs,
            "rxn_lyr": self.rxn_lyr,
            "subnet": self.subnet,
            "output_lyr": self.output_lyr,
            "rxn_norms_exist": self.rxn_norms_exist,
            "rxn_mean": self.rxn_mean,
            "rxn_std_dev": self.rxn_std_dev
        }
        return config

    # from_config doesn't get called anyways?
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def calculate_rxn_normalization(self, 
        rxn_lyr: RxnInputsGaussLayerFV, 
        inputs, 
        percent: float
        ):
        """Calculate reaction normalization. 
            If there are a lot of reactions, this can take a while.
            To speed it up, use a smaller percent value.

        Args:
            rxn_lyr (RxnInputsGaussLayerFV): Reaction input layer to use for the normalization.
            inputs: Inputs to use for the normalization
            percent (float): Percent of the inputs to use for the normalization. Should be in (0,1]
        """
        assert percent > 0
        assert percent <= 1.0

        # Size
        l = len(inputs["mu_v"])
        norm_size = int(percent * l)
        print("Calculating input normalization from: %d samples" % norm_size)
        idxs = np.arange(0,l)
        idxs_subset = np.random.choice(idxs,size=norm_size,replace=False)

        inputs_norm = {}
        for key, val in inputs.items():
            inputs_norm[key] = val[idxs_subset]

        x = rxn_lyr(inputs_norm)
        self.rxn_mean = np.mean(x,axis=0)
        self.rxn_std_dev = np.std(x,axis=0)

        # Correct small
        for i in range(0,len(self.rxn_mean)):
            if abs(self.rxn_mean[i]) < 1e-5:
                self.rxn_mean[i] = 0.0
            if abs(self.rxn_std_dev[i]) < 1e-5:
                self.rxn_std_dev[i] = 1.0

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