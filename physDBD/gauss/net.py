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

        shape = tf.shape(chol_vh)
        batch_size = shape[0]
        nv = shape[2]

        amat = tf.matmul(chol_h, chol_h_t) + tf.matmul(chol_vh, chol_vh_t)
        amat = tf.linalg.inv(amat)
        amat = tf.matmul(chol_vh_t, tf.matmul(amat, chol_vh))
        return tf.eye(nv,batch_shape=[batch_size]) - amat

    def array_flatten_low_tri(self, matv, matvh, math):
        
        shape = tf.shape(matvh)
        batch_size = shape[0]
        nh = shape[1]
        nv = shape[2]

        zeros = tf.zeros((batch_size, nv, nh))
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

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParams0ToParamsGaussLayer(tf.keras.layers.Layer):

    def __init__(self, 
        nv: int, 
        nh: int, 
        layer_muh: Dict[str,FourierLatentGaussLayer],
        layer_cholvh: Dict[str,FourierLatentGaussLayer],
        layer_cholh: Dict[str,FourierLatentGaussLayer],
        **kwargs):

        super(ConvertParams0ToParamsGaussLayer, self).__init__(**kwargs)

        self.nv = nv
        self.nh = nh

        self.layer_muh = layer_muh
        self.layer_cholvh = layer_cholvh
        self.layer_cholh = layer_cholh

        self.convert_from_0 = ConvertParamsGaussLayerFrom0()

    @classmethod
    def construct(cls, 
        nv : int,
        nh : int,
        freqs : np.array,
        muh_sin_coeffs_init : np.array,
        muh_cos_coeffs_init : np.array,
        cholvh_sin_coeffs_init : np.array,
        cholvh_cos_coeffs_init : np.array,
        cholh_sin_coeffs_init : np.array,
        cholh_cos_coeffs_init : np.array,
        **kwargs):

        layer_muh = {}
        for ih in range(0,nh):
            layer_muh[str(ih)] = FourierLatentGaussLayer(
                freqs=freqs,
                offset=0.0,
                sin_coeff=muh_sin_coeffs_init,
                cos_coeff=muh_cos_coeffs_init
                )

        layer_cholvh = {}
        for ih in range(0,nh):
            for iv in range(0,nv):

                layer_cholvh[str(iv) + "_" + str(ih)] = FourierLatentGaussLayer(
                    freqs=freqs,
                    offset=0.0,
                    sin_coeff=cholvh_sin_coeffs_init,
                    cos_coeff=cholvh_cos_coeffs_init
                    )

        layer_cholh = {}
        for ih in range(0,nh):
            for jh in range(0,ih+1):
                layer_cholh[str(ih) + "_" + str(jh)] = FourierLatentGaussLayer(
                    freqs=freqs,
                    offset=0.0,
                    sin_coeff=cholh_sin_coeffs_init,
                    cos_coeff=cholh_cos_coeffs_init
                    )

        return cls(
            nv=nv,
            nh=nh,
            layer_muh=layer_muh,
            layer_cholvh=layer_cholvh,
            layer_cholh=layer_cholh,
            **kwargs
            )

    def get_config(self):
        config = super(ConvertParams0ToParamsGaussLayer, self).get_config()

        config.update({
            "nv": self.nv,
            "nh": self.nh,
            "layer_muh": self.layer_muh,
            "layer_cholvh": self.layer_cholvh,
            "layer_cholh": self.layer_cholh
        })
        
        return config
    
    @classmethod
    def from_config(cls, config):

        layer_muh = {}
        for key, val in config["layer_muh"].items():
            layer_muh[key] = FourierLatentGaussLayer(**val['config'])
        
        layer_cholvh = {}
        for key, val in config["layer_cholvh"].items():
            layer_cholvh[key] = FourierLatentGaussLayer(**val['config'])

        layer_cholh = {}
        for key, val in config["layer_cholh"].items():
            layer_cholh[key] = FourierLatentGaussLayer(**val['config'])

        config["layer_muh"] = layer_muh
        config["layer_cholvh"] = layer_cholvh
        config["layer_cholh"] = layer_cholh

        return cls(**config)
    
    def call(self, inputs):

        batch_size = tf.shape(inputs["tpt"])[0]

        # Get fourier
        muhs = []
        for ih in range(0,self.nh):
            muhs.append(self.layer_muh[str(ih)](inputs))

        cholvhs = []
        for ih in range(0,self.nh):
            for iv in range(0,self.nv):
                cholvhs.append(self.layer_cholvh[str(iv) + "_" + str(ih)](inputs))
        
        cholhs = []
        for ih in range(0,self.nh):
            for jh in range(0,self.nh):
                if jh <= ih:
                    cholhs.append(self.layer_cholh[str(ih) + "_" + str(jh)](inputs))
                else:
                    cholhs.append(tf.zeros(shape=(batch_size)))

        # Current size is (nh, batch_size)
        # Transpose to get (batch_size, nh)
        muh = tf.transpose(muhs)
        cholvh_vec = tf.transpose(cholvhs)
        cholh_vec = tf.transpose(cholhs)

        # To matrix
        cholvh = tf.map_fn(
            lambda cholvh_vecL: tf.reshape(cholvh_vecL, shape=(self.nh,self.nv)), 
            cholvh_vec)
        cholh = tf.map_fn(
            lambda cholh_vecL: tf.reshape(cholh_vecL, shape=(self.nh,self.nh)), 
            cholh_vec)

        inputs_convert = {
            "mu_v1": inputs["mu_v"],
            "chol_v1": inputs["chol_v"],
            "mu_h2": muh,
            "chol_vh2": cholvh,
            "chol_h2": cholh
            }
        outputs_convert = self.convert_from_0(inputs_convert)

        output = {
            "mu": outputs_convert["mu2"],
            "chol": outputs_convert["chol2"]
        }

        return output

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParamsToMomentsGaussLayer(tf.keras.layers.Layer):

    def __init__(self,
        nv : int,
        nh : int,
        **kwargs
        ):
        # Super
        super(ConvertParamsToMomentsGaussLayer, self).__init__(**kwargs)
        self.nv = nv 
        self.nh = nh
    
    def get_config(self):
        config = super(ConvertParamsToMomentsGaussLayer, self).get_config()
        config.update({
            "nv": self.nv,
            "nh": self.nh
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        
        mu = inputs["mu"]
        chol = inputs["chol"]

        cholt = tf.transpose(chol, perm=[0,2,1])

        prec = tf.matmul(chol,cholt)
        cov = tf.linalg.inv(prec)

        return {
            "mu": mu,
            "cov": cov
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertMomentsToNMomentsGaussLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """Convert moments to nMoments
           Moments = (mean, cov_mat)
           nMoments = (mean, n^2 matrix = cov_mat + mean.mean^T)
        """
        # Super
        super(ConvertMomentsToNMomentsGaussLayer, self).__init__(**kwargs)
    
    def get_config(self):
        config = super(ConvertMomentsToNMomentsGaussLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        
        mu = inputs["mu"]
        cov = inputs["cov"]

        # kronecker product of two vectors = tf.tensordot(a,b,axes=0)
        kpv = tf.map_fn(lambda muL: tf.tensordot(muL,muL,axes=0),mu)

        ncov = cov + kpv

        return {
            "mu": mu,
            "ncov": ncov
        }