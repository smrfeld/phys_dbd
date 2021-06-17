from tensorflow.python.ops.gen_math_ops import sub
from .net import RxnInputsLayer
from .params import Params
from .params_traj import ParamsTraj

import tensorflow as tf
import numpy as np

from typing import List

# Do not put this:
# @tf.keras.utils.register_keras_serializable(package="physDBD")

@tf.keras.utils.register_keras_serializable(package="physDBD")
class RxnModel(tf.keras.Model):

    @classmethod
    def construct(cls, 
        nv: int, 
        nh: int, 
        rxn_lyr: RxnInputsLayer, 
        subnet: tf.keras.Model,
        non_zero_outputs : List[str] = [],
        rxn_norms_exist : bool = False,
        rxn_mean : np.array = np.array([]),
        rxn_std_dev : np.array = np.array([]),
        **kwargs
        ):
        """Construct complete TF model with reaction approximations as inputs

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
            rxn_lyr (RxnInputsLayer): Reaction input layer.
            subnet (tf.keras.Model): Subnet; any keras model. Do not put output layer - last layer is automatically placed.
            non_zero_outputs (List[str], optional): List of non-zero outputs in long form.
                E.g. if ["wt00", "b0"], then output of network will just be 2D for these variables. Defaults to [].
            rxn_norms_exist (bool, optional): Flag whether the reaction normalizations exist. 
                Calling calculate_rxn_normalization later constructs them. Defaults to False.
            rxn_mean (np.array, optional): Reaction normalization means. 
                Calling calculate_rxn_normalization later constructs them.. Defaults to np.array([]).
            rxn_std_dev (np.array, optional): Reaction normalization std. devs. 
                Calling calculate_rxn_normalization later constructs them.. Defaults to np.array([]).
        """

        if len(non_zero_outputs) == 0:
            non_zero_outputs_use = []
            for ih in range(0,nh):
                for iv in range(0,nv):
                    non_zero_outputs_use.append("wt%d%d_TE" % (ih,iv))
            for iv in range(0,nv):
                non_zero_outputs_use.append("b%d_TE" % iv)
            non_zero_outputs_use.append("sig2_TE")
        else:
            non_zero_outputs_use = non_zero_outputs
        
        no_outputs = len(non_zero_outputs_use)

        output_lyr = tf.keras.layers.Dense(no_outputs, activation=None)

        return cls(
            nv=nv,
            nh=nh,
            rxn_lyr=rxn_lyr,
            subnet=subnet,
            output_lyr=output_lyr,
            non_zero_outputs=non_zero_outputs_use,
            rxn_norms_exist=rxn_norms_exist,
            rxn_mean=rxn_mean,
            rxn_std_dev=rxn_std_dev
            )

    def __init__(self, 
        nv: int, 
        nh: int,
        rxn_lyr: RxnInputsLayer, 
        subnet: tf.keras.Model,
        output_lyr: tf.keras.layers.Layer,
        non_zero_outputs : List[str],
        rxn_norms_exist : bool = False,
        rxn_mean : np.array = np.array([]),
        rxn_std_dev : np.array = np.array([]),
        **kwargs
        ):
        """Constructor for the complete model with reaction approximations. NOTE: better to use the 'construct' static method helper.

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
            rxn_lyr (RxnInputsLayer): Reaction input layer.
            subnet (tf.keras.Model): Subnet; any keras model. Do not put output layer - last layer is automatically placed.
            output_lyr (tf.keras.layers.Layer): Final output layer.
            non_zero_outputs (List[str], optional): List of non-zero outputs in long form.
                E.g. if ["wt00", "b0"], then output of network will just be 2D for these variables. Defaults to [].
            rxn_norms_exist (bool, optional): Flag whether the reaction normalizations exist. 
                Calling calculate_rxn_normalization later constructs them. Defaults to False.
            rxn_mean (np.array, optional): Reaction normalization means. 
                Calling calculate_rxn_normalization later constructs them.. Defaults to np.array([]).
            rxn_std_dev (np.array, optional): Reaction normalization std. devs. 
                Calling calculate_rxn_normalization later constructs them.. Defaults to np.array([]).
        """
        super(RxnModel, self).__init__(**kwargs)

        self.nv = nv
        self.nh = nh

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
            "nh": self.nh,
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

    def integrate(self, 
        params_start: Params, 
        tpt_start: int, 
        no_steps: int,
        time_interval: float,
        output_mean : np.array,
        output_std_dev : np.array
        ) -> ParamsTraj:
        """Integrate starting from initial params

        Args:
            params_start (Params): Initial params (should be params0 = std. params)
            tpt_start (int): Initial timepoint (index, NOT real time)
            no_steps (int): No. steps to integrate for.
            time_interval (float): Time interval (real time). Only used to construct the list of times in the ParamsTraj returned.
            output_mean (np.array): Output means to undo the normalization.
            output_std_dev (np.array): Output std. devs. to undo the normalization.

        Returns:
            ParamsTraj: Parameter trajectory integrated.
        """

        interval_print = int(no_steps / 10.0)

        tpts_traj = [tpt_start]
        params_traj = [params_start]
        for step in range(0,no_steps):
            if step % interval_print == 0:
                print("%d / %d ..." % (step,no_steps))

            tpt_curr = tpts_traj[-1]
            input0 = params_start.get_tf_input_assuming_params0(tpt=tpt_curr)
            output0 = self.call(input0)

            # Undo normalization on outputs
            ks = list(output0.keys())
            for key in ks:
                output0[key] = output0[key].numpy()[0,0]
                output0[key] *= output_std_dev[key]
                output0[key] += output_mean[key]
            
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

    def calculate_rxn_normalization(self, rxn_lyr: RxnInputsLayer, inputs, percent: float):
        """Calculate reaction normalization

        Args:
            rxn_lyr (RxnInputsLayer): Reaction input layer to use for the normalization.
            inputs: Inputs to use for the normalization
            percent (float): Percent of the inputs to use for the normalization. Should be in (0,1]
        """
        assert percent > 0
        assert percent <= 1.0

        # Size
        l = len(inputs["wt"])
        norm_size = int(percent * l)
        print("Calculating input normalization from: %d samples" % norm_size)
        idxs = np.arange(0,l)
        idxs_subset = np.random.choice(idxs,size=norm_size,replace=False)

        inputs_norm = {}
        for key, val in inputs.items():
            inputs_norm[key] = val[idxs_subset]

        x = rxn_lyr(inputs_norm)
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