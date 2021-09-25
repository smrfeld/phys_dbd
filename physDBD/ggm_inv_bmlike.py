from .net_common import unit_mat_sym
from .helpers import check_non_zero_idx_pairs, construct_mat_non_zero, construct_mat

import tensorflow as tf
import numpy as np
from dataclasses import dataclass

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

@dataclass
class Result:
    target_cov_mat_non_zero : np.array
    trained_model : GGMInvModelBMLike
    init_prec_mat_non_zero : np.array
    init_cov_mat_reconstructed_non_zero : np.array
    learned_prec_mat_non_zero : np.array
    learned_cov_mat_non_zero : np.array

    def report(self):
        print("Prec mat initial guess for non-zero elements:")
        print(self.init_prec_mat_non_zero)

        print("-> Learned prec mat non-zero elements:")
        print(self.learned_prec_mat_non_zero)

        print("Initial cov mat non-zero elements corresponding to initial prec mat guess:")
        print(self.init_cov_mat_reconstructed_non_zero)

        print("-> Learned cov mat non-zero elements:")
        print(self.learned_cov_mat_non_zero)

        print("--> Target cov mat non-zero elements:")
        print(self.target_cov_mat_non_zero)

def invert_ggm_bmlike(
    n: int,
    non_zero_idx_pairs: List[Tuple[int,int]],
    cov_mat_non_zero: np.array,
    epochs: int = 100,
    learning_rate: float = 0.01,
    batch_size = 2
    ) -> Result:

    if batch_size == 1:
        raise ValueError("Batch size = 1 leads to peculiar problems; try anything higher, e.g. 2")
    
    assert(batch_size > 0)

    assert(cov_mat_non_zero.shape == (len(non_zero_idx_pairs),))

    # Invert cov mat to get initial guess for precision matrix
    cov_mat = construct_mat(n,non_zero_idx_pairs,cov_mat_non_zero)
    init_prec_mat = np.linalg.inv(cov_mat)
    init_prec_mat_non_zero = construct_mat_non_zero(n,non_zero_idx_pairs,init_prec_mat)
    print("Prec mat initial guess for non-zero elements", init_prec_mat_non_zero)

    init_prec_mat_reconstructed = construct_mat(n,non_zero_idx_pairs,init_prec_mat_non_zero)
    init_cov_mat_reconstructed = np.linalg.inv(init_prec_mat_reconstructed)
    init_cov_mat_reconstructed_non_zero = construct_mat_non_zero(n,non_zero_idx_pairs,init_cov_mat_reconstructed)
    print("Initial cov mat corresponding non-zero elements", init_cov_mat_reconstructed_non_zero)

    # Make layer and model
    lyr = GGMInvPrecToCovMatLayer(
        n=n,
        non_zero_idx_pairs=non_zero_idx_pairs,
        non_zero_vals=init_prec_mat_non_zero
        )
    model = GGMInvModelBMLike(inv_lyr=lyr)

    # Compile
    loss_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                loss=loss_fn,
                run_eagerly=False)

    # Inputs outputs
    max_batch_size = 100
    inputs = np.full(
        shape=(max_batch_size,1),
        fill_value=np.array([2])
        )

    outputs = np.full(
        shape=(max_batch_size,len(cov_mat_non_zero)),
        fill_value=cov_mat_non_zero
        )
    
    # Train!
    # NOTE: Do NOT use batch_size=1, use anything greater like batch_size=2
    model.fit(
        inputs, 
        outputs, 
        epochs=epochs, 
        batch_size=2
        )

    # Return solution & model
    learned_prec_mat_non_zero = model.inv_lyr.non_zero_vals.numpy()
    learned_prec_mat = construct_mat(n,non_zero_idx_pairs,learned_prec_mat_non_zero)
    learned_cov_mat = np.linalg.inv(learned_prec_mat)
    learned_cov_mat_non_zero = construct_mat_non_zero(n,non_zero_idx_pairs,learned_cov_mat)

    result = Result(
        target_cov_mat_non_zero=cov_mat_non_zero,
        trained_model=model,
        init_prec_mat_non_zero=init_prec_mat_non_zero,
        init_cov_mat_reconstructed_non_zero=init_cov_mat_reconstructed_non_zero,
        learned_prec_mat_non_zero=learned_prec_mat_non_zero,
        learned_cov_mat_non_zero=learned_cov_mat_non_zero
        )

    return result
