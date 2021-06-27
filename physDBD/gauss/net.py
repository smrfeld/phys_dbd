import tensorflow as tf

import numpy as np
from typing import List, Tuple, Union, Dict

@tf.keras.utils.register_keras_serializable(package="physDBD")
class FourierLatentGaussLayer(tf.keras.layers.Layer):

    def __init__(self, 
        freqs : np.array,
        offset : float,
        sin_coeff : np.array,
        cos_coeff : np.array,
        **kwargs
        ):
        """Fourier latent layer

        Args:
            freqs (np.array): 1D arr of frequencies of length L
            offset (float): Float offset
            sin_coeff (np.array): 1D arr of sin coeffs (trained) of length L
            cos_coeff (np.array): 1D arr of sin coeffs (trained) of length L
        """
        super(FourierLatentGaussLayer, self).__init__(**kwargs)

        self.freqs = self.add_weight(
            name="freqs",
            shape=len(freqs),
            trainable=False,
            initializer=tf.constant_initializer(freqs),
            dtype='float32'
            )
        
        self.offset = self.add_weight(
            name="offset",
            shape=1,
            initializer=tf.constant_initializer(offset),
            dtype='float32'
            )

        self.cos_coeff = self.add_weight(
            name="cos_coeff",
            shape=(len(freqs)),
            initializer=tf.constant_initializer(cos_coeff),
            dtype='float32'
            )

        self.sin_coeff = self.add_weight(
            name="sin_coeff",
            shape=(len(freqs)),
            initializer=tf.constant_initializer(sin_coeff),
            dtype='float32'
            )

    def get_config(self):
        config = super(FourierLatentGaussLayer, self).get_config()
        config.update({
            "freqs": self.freqs.numpy(),
            "offset": self.offset.numpy(),
            "cos_coeff": self.cos_coeff.numpy(),
            "sin_coeff": self.sin_coeff.numpy()
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        
        tsin = tf.map_fn(lambda tpt: tf.math.sin(tpt * self.freqs), inputs["tpt"])
        ts = tf.map_fn(lambda tsinL: tf.tensordot(self.sin_coeff, tsinL, 1), tsin)

        tcos = tf.map_fn(lambda tpt: tf.math.cos(tpt * self.freqs), inputs["tpt"])
        tc = tf.map_fn(lambda tcosL: tf.tensordot(self.cos_coeff, tcosL, 1), tcos)

        return self.offset + ts + tc

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParamsGaussLayer(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, **kwargs):
        """Convert params latent space to different muh,varh_diag
        """
        # Super
        super(ConvertParamsGaussLayer, self).__init__(**kwargs)
        
        self.nv = nv
        self.nh = nh

    def get_config(self):
        config = super(ConvertParamsGaussLayer, self).get_config()
        config.update({
            "nv": self.nv,
            "nh": self.nh
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def make_amat(self, chol_vh, chol_h):
        chol_vh_t = tf.transpose(chol_vh,perm=[0,2,1])
        chol_h_t = tf.transpose(chol_h,perm=[0,2,1])

        batch_size = tf.shape(chol_vh)[0]

        amat = tf.matmul(chol_h, chol_h_t) + tf.matmul(chol_vh, chol_vh_t)
        amat = tf.linalg.inv(amat)
        amat = tf.matmul(chol_vh_t, tf.matmul(amat, chol_vh))
        return tf.eye(self.nv,batch_shape=[batch_size]) - amat

    def array_flatten_low_tri(self, matv, matvh, math):
        
        batch_size = tf.shape(matv)[0]
        zeros = tf.zeros((batch_size, self.nv, self.nh))
        upper = tf.concat([matv,zeros],2)
        lower = tf.concat([matvh,math],2)
        mat = tf.concat([upper,lower],1)
        return mat

    def call(self, inputs):
        
        mu1 = inputs["mu1"]
        chol1 = inputs["chol1"]

        mu_h2 = inputs["mu_h2"]
        chol_vh2 = inputs["chol_vh2"]
        chol_h2 = inputs["chol_h2"]

        # Mu
        mu_v1 = mu1[:,:self.nv]
        mu2 = tf.concat([mu_v1,mu_h2],1)

        # Chol
        chol_v1 = chol1[:,:self.nv,:self.nv]
        chol_vh1 = chol1[:,self.nv:,:self.nv]
        chol_h1 = chol1[:,self.nv:,self.nv:]

        # A matrix
        amat1 = self.make_amat(chol_vh1,chol_h1)
        amat2 = self.make_amat(chol_vh2,chol_h2)

        # Chol
        chol_a1 = tf.linalg.cholesky(amat1)
        chol_a2 = tf.linalg.cholesky(amat2)

        # Chol v
        chol_v2 = tf.matmul(chol_v1, tf.matmul(chol_a1, tf.linalg.inv(chol_a2)))

        # Array flatten to new cholesky
        chol2 = self.array_flatten_low_tri(chol_v2, chol_vh2, chol_h2)

        return {
            "mu2": mu2,
            "chol2": chol2
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParamsGaussLayerFrom0(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """Convert params0 = std. params with muh=0, varh_diag=1 to params with different muh,varh_diag
        """
        # Super
        super(ConvertParamsGaussLayerFrom0, self).__init__(**kwargs)

    def get_config(self):
        config = super(ConvertParamsGaussLayerFrom0, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def make_amat(self, chol_vh, chol_h):
        chol_vh_t = tf.transpose(chol_vh,perm=[0,2,1])
        chol_h_t = tf.transpose(chol_h,perm=[0,2,1])

        batch_size = tf.shape(chol_vh)[0]

        amat = tf.matmul(chol_h, chol_h_t) + tf.matmul(chol_vh, chol_vh_t)
        amat = tf.linalg.inv(amat)
        amat = tf.matmul(chol_vh_t, tf.matmul(amat, chol_vh))
        return tf.eye(self.nv,batch_shape=[batch_size]) - amat

    def array_flatten_low_tri(self, matv, matvh, math):
        
        batch_size = tf.shape(matv)[0]
        zeros = tf.zeros((batch_size, self.nv, self.nh))
        upper = tf.concat([matv,zeros],2)
        lower = tf.concat([matvh,math],2)
        mat = tf.concat([upper,lower],1)
        return mat

    def call(self, inputs):
        
        mu_v1 = inputs["mu_v1"]
        chol_v1 = inputs["chol_v1"]

        mu_h2 = inputs["mu_h2"]
        chol_vh2 = inputs["chol_vh2"]
        chol_h2 = inputs["chol_h2"]

        # Concatenatu mu
        mu2 = tf.concat([mu_v1,mu_h2],1)

        # A matrix
        amat2 = self.make_amat(chol_vh2,chol_h2)
        chol_a2 = tf.linalg.cholesky(amat2)

        # Chol v
        chol_v2 = tf.matmul(chol_v1, tf.linalg.inv(chol_a2))

        # Array flatten to new cholesky
        chol2 = self.array_flatten_low_tri(chol_v2, chol_vh2, chol_h2)

        return {
            "mu2": mu2,
            "chol2": chol2
        }
