from .net_common import unit_mat_sym
from .helpers import check_non_zero_idx_pairs, construct_mat, check_symmetric, construct_mat_non_zero

import tensorflow as tf

import numpy as np

from typing import List, Tuple

@tf.keras.utils.register_keras_serializable(package="physDBD")
class GGMmultPrecCovLayer(tf.keras.layers.Layer):

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
        super(GGMmultPrecCovLayer, self).__init__(**kwargs)

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
        config = super(GGMmultPrecCovLayer, self).get_config()
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
        
        prec_mat = tf.zeros_like(inputs["cov_mat"])
        for i,pair in enumerate(self.non_zero_idx_pairs):
            prec_mat_0 = self.non_zero_vals[i] * unit_mat_sym(self.n,pair[0],pair[1])
            prec_mat += tf.map_fn(lambda val: prec_mat_0, prec_mat)

        return tf.matmul(prec_mat,inputs["cov_mat"])

@tf.keras.utils.register_keras_serializable(package="physDBD")
class GGMInvModel(tf.keras.Model):

    def __init__(self, 
        mult_lyr: GGMmultPrecCovLayer,
        **kwargs
        ):
        super(GGMInvModel, self).__init__(**kwargs)

        self.mult_lyr = mult_lyr

    def get_config(self):
        # Do not call super.get_config !!!
        config = {
            "mult_lyr": self.mult_lyr
        }
        return config

    # from_config doesn't get called anyways?
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input_tensor, training=False):
        return self.mult_lyr(input_tensor)

def invert_ggm(
    n: int,
    non_zero_idx_pairs: List[Tuple[int,int]],
    cov_mat: np.array,
    epochs: int = 1000,
    learning_rate: float = 0.1,
    batch_size : int = 2
    ) -> Tuple[np.array,float]:

    if batch_size == 1:
        raise ValueError("Batch size = 1 leads to peculiar problems; try anything higher, e.g. 2")
    
    assert(batch_size > 0)

    assert(cov_mat.shape == (n,n))
    assert(check_symmetric(cov_mat))
    
    # Invert 
    prec_mat_init = np.linalg.inv(cov_mat)
    prec_mat_non_zero_init = construct_mat_non_zero(n,non_zero_idx_pairs,prec_mat_init)
    print("here:")
    print(prec_mat_non_zero_init)

    lyr = GGMmultPrecCovLayer(
        n=n,
        non_zero_idx_pairs=non_zero_idx_pairs,
        non_zero_vals=prec_mat_non_zero_init
        )

    model = GGMInvModel(mult_lyr=lyr)

    loss_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                loss=loss_fn,
                run_eagerly=False)

    # Covariance matrix input
    inputs = {"cov_mat": [cov_mat]}

    # Output = identity
    outputs = [np.eye(n)]

    # Train!
    # Do NOT pass batch_size = 1 -> peculiar problems
    tf.get_logger().setLevel('ERROR')
    model.fit(
        inputs, 
        outputs, 
        epochs=epochs, 
        batch_size=20
        )

    # Return
    final_loss = loss_fn(outputs, model(inputs)).numpy()
    return (model.mult_lyr.non_zero_vals.numpy(),final_loss)

def invert_ggm_chol(
    n: int,
    non_zero_idx_pairs: List[Tuple[int,int]],
    cov_mat: np.array,
    epochs: int = 1000,
    learning_rate: float = 0.1,
    batch_size : int = 2
    ) -> Tuple[np.array,float]:

    prec_mat_non_zero, final_loss = invert_ggm(
        n=n,
        non_zero_idx_pairs=non_zero_idx_pairs,
        cov_mat=cov_mat,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
        )

    prec_mat = construct_mat(n,non_zero_idx_pairs,prec_mat_non_zero)

    chol = np.linalg.cholesky(prec_mat)
    return (chol, final_loss)