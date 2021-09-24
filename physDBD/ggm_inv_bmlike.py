from .net_common import unit_mat_sym
from .helpers import check_non_zero_idx_pairs, construct_mat_non_zero

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
        check_non_zero_idx_pairs(n, non_zero_idx_pairs)

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

        check_non_zero_idx_pairs(n, non_zero_idx_pairs)

        self.n = n
        self.non_zero_idx_pairs = non_zero_idx_pairs

        self.non_zero_vals = self.add_weight(
            name="non_zero_vals",
            shape=len(non_zero_vals),
            trainable=True,
            initializer=tf.constant_initializer(non_zero_vals),
            dtype='float32'
            )

    @property
    def n_non_zero(self):
        return len(self.non_zero_idx_pairs)

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
        
        batch_size = tf.shape(inputs)[0]
        print("batch_size:", batch_size)
        prec_mat = tf.zeros((batch_size,self.n,self.n), dtype='float32')

        for i,pair in enumerate(self.non_zero_idx_pairs):
            prec_mat += tf.map_fn(lambda val: self.non_zero_vals[i] * unit_mat_sym(self.n,pair[0],pair[1]), prec_mat)

        print("prec_mat",prec_mat)
        
        cov_mat = tf.linalg.inv(prec_mat)
        print("cov_mat", cov_mat)

        return cov_mat

@tf.keras.utils.register_keras_serializable(package="physDBD")
class GGMInvModelBMLike(tf.keras.Model):

    def __init__(self, 
        inv_lyr: GGMInvPrecToCovMatLayer,
        **kwargs
        ):
        super(GGMInvModelBMLike, self).__init__(**kwargs)

        self.inv_lyr = inv_lyr

    def get_config(self):
        # Do not call super.get_config !!!
        config = {
            "inv_lyr": self.inv_lyr
        }
        return config

    # from_config doesn't get called anyways?
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input_tensor, training=False):
        cov_mat = self.inv_lyr(input_tensor)

        batch_size = tf.shape(cov_mat)[0]
        cov_mat_non_zero = tf.zeros((batch_size,self.inv_lyr.n_non_zero), dtype='float32')

        for i,pair in enumerate(self.inv_lyr.non_zero_idx_pairs):
            unit_vec = tf.one_hot(
                indices=i,
                depth=self.inv_lyr.n_non_zero,
                dtype='float32'
                )
            cov_mat_non_zero += tf.map_fn(lambda cov_mat_batch: cov_mat_batch[pair[0],pair[1]] * unit_vec, cov_mat)

        print("cov_mat_non_zero",cov_mat_non_zero)

        return cov_mat_non_zero

def invert_ggm_bmlike(
    n: int,
    non_zero_idx_pairs: List[Tuple[int,int]],
    cov_mat: np.array,
    epochs: int,
    learning_rate: float = 0.01
    ) -> np.array:

    # Invert 
    prec_mat_init = np.linalg.inv(cov_mat)
    prec_mat_non_zero_init = construct_mat_non_zero(n,non_zero_idx_pairs,prec_mat_init)

    lyr = GGMInvPrecToCovMatLayer(
        n=n,
        non_zero_idx_pairs=non_zero_idx_pairs,
        non_zero_vals=prec_mat_non_zero_init
        )

    model = GGMInvModelBMLike(inv_lyr=lyr)

    loss_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                loss=loss_fn,
                run_eagerly=False)

    inputs = [{}]

    cov_mat_non_zero = construct_mat_non_zero(n,non_zero_idx_pairs,cov_mat)
    outputs = [cov_mat_non_zero]

    # Train!
    model.fit(
        inputs, 
        outputs, 
        epochs=epochs, 
        batch_size=1
        )

    # Return
    return model.inv_lyr.non_zero_vals.numpy()
