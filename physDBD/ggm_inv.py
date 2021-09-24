from .net_common import unit_mat_sym

import tensorflow as tf
import numpy as np

from typing import List, Tuple

@tf.keras.utils.register_keras_serializable(package="physDBD")
class GGMInvPrecToCovMatLayer(tf.keras.layers.Layer):

    @classmethod
    def construct(cls,
        n: int,
        non_zero_idx_pairs: List[Tuple[int,int]],
        init_diag_val: float = 1.0,
        **kwargs
        ):

        # Set all diagonal elements to one, rest zero (like a identity matrix)
        non_zero_vals = np.zeros(len(non_zero_idx_pairs))
        for i,pair in enumerate(non_zero_idx_pairs):
            if pair[0] == pair[1]:
                non_zero_vals[i] = init_diag_val

        return cls(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            non_zero_vals=non_zero_vals,
            **kwargs
            )

    def __init__(self,
        n: int,
        non_zero_idx_pairs: List[Tuple[int,int]],
        non_zero_vals: np.array,
        **kwargs
        ):
        super(GGMInvPrecToCovMatLayer, self).__init__(**kwargs)

        self.n = n
        self.non_zero_idx_pairs = non_zero_idx_pairs

        self.non_zero_vals = self.add_weight(
            name="non_zero_vals",
            shape=len(non_zero_vals),
            trainable=True,
            initializer=tf.constant_initializer(non_zero_vals),
            dtype='float32'
            )

    def get_config(self):
        config = super(GGMInvPrecToCovMatLayer, self).get_config()
        config.update({
            "n": self.n,
            "non_zero_idx_pairs": self.non_zero_idx_pairs,
            "non_zero_vals": self.non_zero_vals.numpy()
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        
        prec_mat = tf.zeros((self.n,self.n))

        for i,pair in enumerate(self.non_zero_idx_pairs):
            prec_mat += self.non_zero_vals[i] * unit_mat_sym(self.n,pair[0],pair[1])

        return tf.linalg.inv(prec_mat)

'''
@tf.keras.utils.register_keras_serializable(package="physDBD")
class GGMInvModel(tf.keras.Model):

    @classmethod
    def construct(cls, 
        n: int, 
        non_zero_idx_pairs: List[Tuple[int,int]],
        **kwargs
        ):

        for i in range(0,n):
            if not (i,i) in non_zero_idx_pairs:
                raise ValueError("All diagonal elements must be specified as non-zero.")

        for pair in non_zero_idx_pairs:
            if pair[0] < pair[1]:
                raise ValueError("Only provide lower triangular indexes.")

        return cls(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            **kwargs
            )

    def __init__(self, 
        n: int, 
        non_zero_idx_pairs: List[Tuple[int,int]],
        **kwargs
        ):
        super(GGMInvModel, self).__init__(**kwargs)

        self.n = n
        self.non_zero_idx_pairs = non_zero_idx_pairs

        # Values

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
            input0 = params_traj[-1].get_tf_input_assuming_params0(tpt=tpt_curr)
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
'''