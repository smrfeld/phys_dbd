from ..net_common import unit_mat, BirthRxnLayer, DeathRxnLayer, \
    EatRxnLayer, ConvertMomentsToNMomentsLayer, ConvertNMomentsTEtoMomentsTE

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
    
    def make_amat(self, chol_hv, chol_h):
        chol_hv_t = tf.transpose(chol_hv,perm=[0,2,1])
        chol_h_t = tf.transpose(chol_h,perm=[0,2,1])

        batch_size = tf.shape(chol_hv)[0]

        amat = tf.matmul(chol_h, chol_h_t) + tf.matmul(chol_hv, chol_hv_t)
        amat = tf.linalg.inv(amat)
        amat = tf.matmul(chol_hv_t, tf.matmul(amat, chol_hv))
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
        chol_hv2 = inputs["chol_hv2"]
        chol_h2 = inputs["chol_h2"]

        # Mu
        mu_v1 = mu1[:,:self.nv]
        mu2 = tf.concat([mu_v1,mu_h2],1)

        # Chol
        chol_v1 = chol1[:,:self.nv,:self.nv]
        chol_hv1 = chol1[:,self.nv:,:self.nv]
        chol_h1 = chol1[:,self.nv:,self.nv:]

        # A matrix
        amat1 = self.make_amat(chol_hv1,chol_h1)
        amat2 = self.make_amat(chol_hv2,chol_h2)

        # Chol
        chol_a1 = tf.linalg.cholesky(amat1)
        chol_a2 = tf.linalg.cholesky(amat2)

        # Chol v
        chol_v2 = tf.matmul(chol_v1, tf.matmul(chol_a1, tf.linalg.inv(chol_a2)))

        # Array flatten to new cholesky
        chol2 = self.array_flatten_low_tri(chol_v2, chol_hv2, chol_h2)

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
    
    def make_amat(self, chol_hv, chol_h):
        chol_hv_t = tf.transpose(chol_hv,perm=[0,2,1])
        chol_h_t = tf.transpose(chol_h,perm=[0,2,1])

        shape = tf.shape(chol_hv)
        batch_size = shape[0]
        nv = shape[2]

        amat = tf.matmul(chol_h, chol_h_t) + tf.matmul(chol_hv, chol_hv_t)
        amat = tf.linalg.inv(amat)
        amat = tf.matmul(chol_hv_t, tf.matmul(amat, chol_hv))
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
        chol_hv2 = inputs["chol_hv2"]
        chol_h2 = inputs["chol_h2"]

        # Concatenatu mu
        mu2 = tf.concat([mu_v1,mu_h2],1)

        # A matrix
        amat2 = self.make_amat(chol_hv2,chol_h2)
        chol_a2 = tf.linalg.cholesky(amat2)

        # Chol v
        chol_v2 = tf.matmul(chol_v1, tf.linalg.inv(chol_a2))

        # Array flatten to new cholesky
        chol2 = self.array_flatten_low_tri(chol_v2, chol_hv2, chol_h2)

        return {
            "mu2": mu2,
            "chol2": chol2,
            "chol_a2": chol_a2
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParams0ToParamsGaussLayer(tf.keras.layers.Layer):

    def __init__(self, 
        nv: int, 
        nh: int, 
        layer_muh: Dict[str,FourierLatentGaussLayer],
        layer_cholhv: Dict[str,FourierLatentGaussLayer],
        layer_cholh: Dict[str,FourierLatentGaussLayer],
        non_zero_idx_pairs_vv: List[Tuple[int,int]],
        non_zero_idx_pairs_hv: List[Tuple[int,int]],
        non_zero_idx_pairs_hh: List[Tuple[int,int]],
        **kwargs):

        super(ConvertParams0ToParamsGaussLayer, self).__init__(**kwargs)

        # Check non-zero idx pairs
        for iv in range(0,nv):
            if not (iv,iv) in non_zero_idx_pairs_vv:
                raise ValueError("All diagonal elements must be specified as non-zero.")
        for ih in range(0,nh):
            if not (ih,ih) in non_zero_idx_pairs_hh:
                raise ValueError("All diagonal elements must be specified as non-zero.")

        for iv,jv in non_zero_idx_pairs_vv:
            if iv < jv:
                raise ValueError("Only provide lower triangular indexes.")
            if (not iv < nv) or (not jv < nv):
                raise ValueError("Indexes in non_zero_idx_pairs_vv must be in [0,nv)")

        for ih,jh in non_zero_idx_pairs_hh:
            if ih < jh:
                raise ValueError("Only provide lower triangular indexes.")
            if (not ih < nh) or (not jh < nh):
                raise ValueError("Indexes in non_zero_idx_pairs_hh must be in [0,nh)")

        for ih,iv in non_zero_idx_pairs_hv:
            if (not ih < nh) or (not iv < nv):
                raise ValueError("Indexes in non_zero_idx_pairs_hv must be in [(0,0),(nh,nv))")

        self.nv = nv
        self.nh = nh

        self.layer_muh = layer_muh
        self.layer_cholhv = layer_cholhv
        self.layer_cholh = layer_cholh
        
        self.non_zero_idx_pairs_vv = non_zero_idx_pairs_vv
        self.non_zero_idx_pairs_hv = non_zero_idx_pairs_hv
        self.non_zero_idx_pairs_hh = non_zero_idx_pairs_hh

        self.convert_from_0 = ConvertParamsGaussLayerFrom0()

    @classmethod
    def construct(cls, 
        nv : int,
        nh : int,
        freqs : np.array,
        muh_sin_coeffs_init : np.array,
        muh_cos_coeffs_init : np.array,
        cholhv_sin_coeffs_init : np.array,
        cholhv_cos_coeffs_init : np.array,
        cholh_sin_coeffs_init : np.array,
        cholh_cos_coeffs_init : np.array,
        non_zero_idx_pairs_vv: List[Tuple[int,int]],
        non_zero_idx_pairs_hv: List[Tuple[int,int]],
        non_zero_idx_pairs_hh: List[Tuple[int,int]],
        **kwargs):

        layer_muh = {}
        for ih in range(0,nh):
            layer_muh[str(ih)] = FourierLatentGaussLayer(
                freqs=freqs,
                offset=0.0,
                sin_coeff=muh_sin_coeffs_init,
                cos_coeff=muh_cos_coeffs_init
                )

        layer_cholhv = {}
        for ih in range(0,nh):
            for iv in range(0,nv):
                if (ih,iv) in non_zero_idx_pairs_hv:
                    layer_cholhv[str(ih) + "_" + str(iv)] = FourierLatentGaussLayer(
                        freqs=freqs,
                        offset=0.0,
                        sin_coeff=cholhv_sin_coeffs_init,
                        cos_coeff=cholhv_cos_coeffs_init
                        )

        layer_cholh = {}
        for ih in range(0,nh):
            for jh in range(0,ih+1):
                if (ih,jh) in non_zero_idx_pairs_hh:
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
            layer_cholhv=layer_cholhv,
            layer_cholh=layer_cholh,
            non_zero_idx_pairs_vv=non_zero_idx_pairs_vv,
            non_zero_idx_pairs_hv=non_zero_idx_pairs_hv,
            non_zero_idx_pairs_hh=non_zero_idx_pairs_hh,
            **kwargs
            )

    def get_config(self):
        config = super(ConvertParams0ToParamsGaussLayer, self).get_config()

        config.update({
            "nv": self.nv,
            "nh": self.nh,
            "layer_muh": self.layer_muh,
            "layer_cholhv": self.layer_cholhv,
            "layer_cholh": self.layer_cholh,
            "non_zero_idx_pairs_vv": self.non_zero_idx_pairs_vv,
            "non_zero_idx_pairs_hv": self.non_zero_idx_pairs_hv,
            "non_zero_idx_pairs_hh": self.non_zero_idx_pairs_hh
        })
        
        return config
    
    @classmethod
    def from_config(cls, config):

        layer_muh = {}
        for key, val in config["layer_muh"].items():
            layer_muh[key] = FourierLatentGaussLayer(**val['config'])
        
        layer_cholhv = {}
        for key, val in config["layer_cholhv"].items():
            layer_cholhv[key] = FourierLatentGaussLayer(**val['config'])

        layer_cholh = {}
        for key, val in config["layer_cholh"].items():
            layer_cholh[key] = FourierLatentGaussLayer(**val['config'])

        config["layer_muh"] = layer_muh
        config["layer_cholhv"] = layer_cholhv
        config["layer_cholh"] = layer_cholh

        return cls(**config)
    
    def call(self, inputs):

        batch_size = tf.shape(inputs["tpt"])[0]

        # Get fourier
        muhs = []
        for ih in range(0,self.nh):
            muhs.append(self.layer_muh[str(ih)](inputs))

        cholhvs = []
        for ih in range(0,self.nh):
            for iv in range(0,self.nv):
                if (ih,iv) in self.non_zero_idx_pairs_hv:
                    cholhvs.append(self.layer_cholhv[str(ih) + "_" + str(iv)](inputs))
                else:
                    cholhvs.append(tf.zeros(shape=(batch_size)))
        
        cholhs = []
        for ih in range(0,self.nh):
            for jh in range(0,self.nh):
                if jh <= ih and (ih,jh) in self.non_zero_idx_pairs_hh:
                    cholhs.append(self.layer_cholh[str(ih) + "_" + str(jh)](inputs))
                else:
                    cholhs.append(tf.zeros(shape=(batch_size)))

        # Current size is (nh, batch_size)
        # Transpose to get (batch_size, nh)
        mu_h2 = tf.transpose(muhs)
        cholhv_vec = tf.transpose(cholhvs)
        cholh_vec = tf.transpose(cholhs)

        # To matrix
        chol_hv2 = tf.map_fn(
            lambda cholhv_vecL: tf.reshape(cholhv_vecL, shape=(self.nh,self.nv)), 
            cholhv_vec)
        chol_h2 = tf.map_fn(
            lambda cholh_vecL: tf.reshape(cholh_vecL, shape=(self.nh,self.nh)), 
            cholh_vec)

        # Ensure structure in chol v
        chol_v1 = tf.zeros((batch_size,self.nv,self.nv))
        chol_v_non_zero = inputs["chol_v_non_zero"]
        for idx in range(0,len(self.non_zero_idx_pairs_vv)):
            iv,jv = self.non_zero_idx_pairs_vv[idx]
            chol_v1 += tf.map_fn(
                        lambda chol_v_non_zeroUP: unit_mat(self.nv,iv,jv) * chol_v_non_zeroUP[idx],
                        chol_v_non_zero)

        inputs_convert = {
            "mu_v1": inputs["mu_v"],
            "chol_v1": chol_v1,
            "mu_h2": mu_h2,
            "chol_hv2": chol_hv2,
            "chol_h2": chol_h2
            }
        outputs_convert = self.convert_from_0(inputs_convert)

        output = {
            "mu": outputs_convert["mu2"],
            "chol": outputs_convert["chol2"],
            "chol_amat": outputs_convert["chol_a2"]
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
        prec_h = prec[:,self.nv:,self.nv:]
        cov = tf.linalg.inv(prec)

        return {
            "prec": prec,
            "prec_h": prec_h,
            "mu": mu,
            "cov": cov
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParams0ToNMomentsGaussLayer(tf.keras.layers.Layer):

    def __init__(self, 
        nv: int,
        nh: int,
        params0ToParamsLayer: ConvertParams0ToParamsGaussLayer,
        **kwargs
        ):
        """Convert params0 to NMoments layer in one step
           Params0 = std. params with muh=0,varh_diag=1
           nMoments = (mean, n^2 matrix = cov_mat + mean.mean^T)

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
            params0ToParamsLayer (ConvertParams0ToParamsGaussLayer): Conversion layer from params0 to params via latent Fourier representation
        """
        # Super
        super(ConvertParams0ToNMomentsGaussLayer, self).__init__(**kwargs)

        self.nv = nv
        self.nh = nh

        self.params0ToParamsLayer = params0ToParamsLayer
        self.paramsToMomentsLayer = ConvertParamsToMomentsGaussLayer(nv,nh)
        self.momentsToNMomentsLayer = ConvertMomentsToNMomentsLayer()

    @classmethod
    def construct(cls,
        nv: int, 
        nh: int,
        freqs : np.array,
        muh_sin_coeffs_init : np.array,
        muh_cos_coeffs_init : np.array,
        cholhv_sin_coeffs_init : np.array,
        cholhv_cos_coeffs_init : np.array,
        cholh_sin_coeffs_init : np.array,
        cholh_cos_coeffs_init : np.array,
        non_zero_idx_pairs_vv: List[Tuple[int,int]],
        non_zero_idx_pairs_hv: List[Tuple[int,int]],
        non_zero_idx_pairs_hh: List[Tuple[int,int]],
        **kwargs
        ):
        params0ToParamsLayer = ConvertParams0ToParamsGaussLayer.construct(
            nv=nv, 
            nh=nh, 
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            cholhv_sin_coeffs_init=cholhv_sin_coeffs_init,
            cholhv_cos_coeffs_init=cholhv_cos_coeffs_init,
            cholh_sin_coeffs_init=cholh_sin_coeffs_init,
            cholh_cos_coeffs_init=cholh_cos_coeffs_init,
            non_zero_idx_pairs_vv=non_zero_idx_pairs_vv,
            non_zero_idx_pairs_hv=non_zero_idx_pairs_hv,
            non_zero_idx_pairs_hh=non_zero_idx_pairs_hh
            )

        return cls(
            nv=nv,
            nh=nh,
            params0ToParamsLayer=params0ToParamsLayer,
            **kwargs
            )

    def get_config(self):
        config = super(ConvertParams0ToNMomentsGaussLayer, self).get_config()
        config.update({
            "nv": self.nv,
            "nh": self.nh,
            "params0ToParamsLayer": self.params0ToParamsLayer
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        params = self.params0ToParamsLayer(inputs)
        moments = self.paramsToMomentsLayer(params)
        nmoments = self.momentsToNMomentsLayer(moments)

        # Collect all parts
        dall = {}
        for d in [params,moments,nmoments]:
            dall.update(d)
        
        return dall

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertMomentsTEtoParamsTEGaussLayer(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, **kwargs):
        # Super
        super(ConvertMomentsTEtoParamsTEGaussLayer, self).__init__(**kwargs)
    
        self.nv = nv
        self.nh = nh

    def get_config(self):
        config = super(ConvertMomentsTEtoParamsTEGaussLayer, self).get_config()
        config.update({
            "nv": self.nv,
            "nh": self.nh
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        muTE = inputs["muTE"]
        covTE = inputs["covTE"]

        chol = inputs["chol"]
        cholT = tf.transpose(chol,perm=[0,2,1])

        batch_size = tf.shape(inputs["muTE"])[0]

        phi_mat = tf.matmul(cholT,tf.matmul(covTE,chol))

        n = self.nv + self.nh
        cholTE = -1.0 * tf.matmul(chol, apply_phi_mat(phi_mat, n, batch_size))

        return {
            "muTE": muTE,
            "cholTE": cholTE
        }

def apply_phi_mat(arg_mat, n: int, batch_size: int):
    res = tf.zeros((batch_size,n,n))
    for i in range(0,n):
        for j in range(0,n):
            if i > j:
                res += tf.map_fn(
                    lambda arg_matL: unit_mat(n,i,j)*arg_matL[i,j],
                    arg_mat)
            elif i == j:
                res += tf.map_fn(
                    lambda arg_matL: 0.5*unit_mat(n,i,j)*arg_matL[i,j],
                    arg_mat)
    return res

def make_chol_mat_TE(matTE, chol_mat, n: int, batch_size: int):
    chol_mat_inv = tf.linalg.inv(chol_mat)
    chol_mat_inv_T = tf.transpose(chol_mat_inv,perm=[0,2,1])
    arg_mat = tf.matmul(chol_mat_inv, tf.matmul(matTE, chol_mat_inv_T))
    phi_mat = apply_phi_mat(arg_mat, n, batch_size)
    return tf.matmul(chol_mat, phi_mat)

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParamsTEtoParams0TEGaussLayer(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, **kwargs):
        # Super
        super(ConvertParamsTEtoParams0TEGaussLayer, self).__init__(**kwargs)

        self.nv = nv
        self.nh = nh

    def get_config(self):
        config = super(ConvertParamsTEtoParams0TEGaussLayer, self).get_config()
        config.update({
            "nv": self.nv,
            "nh": self.nh
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs):

        # Inputs in the non-std space
        mu_TE = inputs["mu_TE"]

        batch_size = tf.shape(inputs["mu_TE"])[0]

        prec_h = inputs["prec_h"]
        prec_h_inv = tf.linalg.inv(prec_h)

        chol_v = inputs["chol_v"]
        chol_h = inputs["chol_h"]
        chol_hv = inputs["chol_hv"]

        chol_v_TE = inputs["chol_v_TE"]
        chol_h_TE = inputs["chol_h_TE"]
        chol_hv_TE = inputs["chol_hv_TE"]

        chol_h_T = tf.transpose(chol_h,perm=[0,2,1])
        chol_hv_T = tf.transpose(chol_hv,perm=[0,2,1])

        chol_h_TE_T = tf.transpose(chol_h_TE,perm=[0,2,1])
        chol_hv_TE_T = tf.transpose(chol_hv_TE,perm=[0,2,1])

        prec_h_inv_TE_int = tf.matmul(chol_h_TE, chol_h_T) \
            + tf.matmul(chol_h, chol_h_TE_T) \
            + tf.matmul(chol_hv_TE, chol_hv_T) \
            + tf.matmul(chol_hv, chol_hv_TE_T)
        prec_h_inv_TE = - tf.matmul(prec_h_inv, tf.matmul(prec_h_inv_TE_int, prec_h_inv))

        amat_TE = - tf.matmul(chol_hv_TE_T, tf.matmul(prec_h_inv, chol_hv))
        amat_TE += - tf.matmul(chol_hv_T, tf.matmul(prec_h_inv_TE, chol_hv))
        amat_TE += - tf.matmul(chol_hv_T, tf.matmul(prec_h_inv, chol_hv_TE))

        chol_amat = inputs["chol_amat"]
        chol_amat_TE = make_chol_mat_TE(amat_TE, chol_amat, self.nv, batch_size)
        
        cholv_TE_std = tf.matmul(chol_v_TE,chol_amat) + tf.matmul(chol_v, chol_amat_TE)
        muv_TE = mu_TE

        return {
            "cholv_TE_std": cholv_TE_std,
            "muv_TE": mu_TE[:,:self.nv]
            }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertNMomentsTEtoParams0TEGaussLayer(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, **kwargs):

        # Super
        super(ConvertNMomentsTEtoParams0TEGaussLayer, self).__init__(**kwargs)
        
        self.nv = nv
        self.nh = nh

        self.nMomentsTEtoMomentsTE = ConvertNMomentsTEtoMomentsTE()
        self.momentsTEtoParamsTE = ConvertMomentsTEtoParamsTEGaussLayer(nv,nh)
        self.paramsTEtoParams0TE = ConvertParamsTEtoParams0TEGaussLayer(nv,nh)

    def get_config(self):
        config = super(ConvertNMomentsTEtoParams0TEGaussLayer, self).get_config()
        config.update({
            "nv": self.nv,
            "nh": self.nh
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        outputs0 = self.nMomentsTEtoMomentsTE(inputs)

        outputs0.update(inputs)
        outputs1 = self.momentsTEtoParamsTE(outputs0)

        chol_TE = outputs1["cholTE"]
        chol_v_TE = chol_TE[:,:self.nv,:self.nv]
        chol_h_TE = chol_TE[:,self.nv:,self.nv:]
        chol_hv_TE = chol_TE[:,self.nv:,:self.nv]

        chol = inputs["chol"]
        chol_v = chol[:,:self.nv,:self.nv]
        chol_h = chol[:,self.nv:,self.nv:]
        chol_hv = chol[:,self.nv:,:self.nv]

        inputs2 = {
            "mu_TE": outputs1["muTE"],
            "prec_h": inputs["prec_h"],
            "chol_v": chol_v,
            "chol_h": chol_h,
            "chol_hv": chol_hv,
            "chol_v_TE": chol_v_TE,
            "chol_h_TE": chol_h_TE,
            "chol_hv_TE": chol_hv_TE,
            "chol_amat": inputs["chol_amat"]
            }

        outputs2 = self.paramsTEtoParams0TE(inputs2)
        return outputs2

@tf.keras.utils.register_keras_serializable(package="physDBD")
class RxnInputsGaussLayer(tf.keras.layers.Layer):

    def __init__(self, 
        nv: int, 
        nh: int, 
        params0toMoments: ConvertParams0ToNMomentsGaussLayer,
        rxn_specs : List[Union[Tuple[str,int],Tuple[str,int,int]]],
        **kwargs
        ):
        """Calculate inputs from different reaction approximations in a single layer. 

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
            params0toMoments (Convertparams0toMomentsLayer): Layer that converts params0 to moments
            rxn_specs (List[Union[Tuple[str,int],Tuple[str,int,int]]]): List of reaction specifications of tuples.
                First arguments is one of "BIRTH", "DEATH" or "EAT"
                Second/third argument are the index of the species undergoing the reaction
                e.g. ("EAT",3,2) is the Predator-Prey reaction for species #3 being the huner, species #2 the prey

        Raises:
            ValueError: If reaaction string is not recognized
        """
        # Super
        super(RxnInputsGaussLayer, self).__init__(**kwargs)
        
        self.nv = nv
        self.nh = nh

        self.params0toMoments = params0toMoments
        self.momentsTEtoParams0TE = ConvertNMomentsTEtoParams0TEGaussLayer(nv=nv,nh=nh)

        self.rxn_specs = rxn_specs
        self.rxns = []
        for spec in rxn_specs:
            if spec[0] == "EAT":
                rtype, i_hunter, i_prey = spec
            else:
                rtype, i_sp = spec

            if rtype == "BIRTH":
                self.rxns.append(BirthRxnLayer(nv, nh, i_sp))
            elif rtype == "DEATH":
                self.rxns.append(DeathRxnLayer(nv, nh, i_sp))
            elif rtype == "EAT":
                self.rxns.append(EatRxnLayer(nv,nh,i_hunter,i_prey))
            else:
                raise ValueError("Rxn type: %s not recognized" % rtype)
    
    @classmethod
    def construct_zero_init(cls, 
        nv: int, 
        nh: int, 
        freqs : np.array,
        non_zero_idx_pairs_vv: List[Tuple[int,int]],
        non_zero_idx_pairs_hv: List[Tuple[int,int]],
        non_zero_idx_pairs_hh: List[Tuple[int,int]],
        rxn_specs : List[Union[Tuple[str,int],Tuple[str,int,int]]],
        **kwargs
        ):
        return cls.construct(
            nv=nv,
            nh=nh,
            freqs=freqs,
            muh_sin_coeffs_init=np.full(len(freqs),0.0),
            muh_cos_coeffs_init=np.full(len(freqs),0.0),
            cholhv_sin_coeffs_init=np.full(len(freqs),0.0),
            cholhv_cos_coeffs_init=np.full(len(freqs),0.0),
            cholh_sin_coeffs_init=np.full(len(freqs),0.0),
            cholh_cos_coeffs_init=np.full(len(freqs),0.0),
            non_zero_idx_pairs_vv=non_zero_idx_pairs_vv,
            non_zero_idx_pairs_hv=non_zero_idx_pairs_hv,
            non_zero_idx_pairs_hh=non_zero_idx_pairs_hh,
            rxn_specs=rxn_specs
            )

    @classmethod
    def construct(cls, 
        nv: int, 
        nh: int,
        freqs : np.array,
        muh_sin_coeffs_init : np.array,
        muh_cos_coeffs_init : np.array,
        cholhv_sin_coeffs_init : np.array,
        cholhv_cos_coeffs_init : np.array,
        cholh_sin_coeffs_init : np.array,
        cholh_cos_coeffs_init : np.array,
        non_zero_idx_pairs_vv: List[Tuple[int,int]],
        non_zero_idx_pairs_hv: List[Tuple[int,int]],
        non_zero_idx_pairs_hh: List[Tuple[int,int]],
        rxn_specs : List[Union[Tuple[str,int],Tuple[str,int,int]]],
        **kwargs
        ):
        
        params0toMoments = ConvertParams0ToNMomentsGaussLayer.construct(
            nv=nv,
            nh=nh,
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            cholhv_sin_coeffs_init=cholhv_sin_coeffs_init,
            cholhv_cos_coeffs_init=cholhv_cos_coeffs_init,
            cholh_sin_coeffs_init=cholh_sin_coeffs_init,
            cholh_cos_coeffs_init=cholh_cos_coeffs_init,
            non_zero_idx_pairs_vv=non_zero_idx_pairs_vv,
            non_zero_idx_pairs_hv=non_zero_idx_pairs_hv,
            non_zero_idx_pairs_hh=non_zero_idx_pairs_hh
            )

        return cls(
            nv=nv,
            nh=nh,
            params0toMoments=params0toMoments,
            rxn_specs=rxn_specs,
            **kwargs
            )

    def get_config(self):
        config = super(RxnInputsGaussLayer, self).get_config()

        config.update({
            "nv": self.nv,
            "nh": self.nh,
            "params0toMoments": self.params0toMoments,
            "rxn_specs": self.rxn_specs
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def make_tril_bool_mask(self, batch_size):
        mask = np.tril(np.ones((self.nv, self.nv), dtype=np.bool_))
        # Repeat mask into batch size
        # Size then is (batch_size,nv,nv)
        mask = np.repeat(mask[np.newaxis, :, :], batch_size, axis=0)
        return mask

    def call(self, inputs):
        moments = self.params0toMoments(inputs)
        batch_size = tf.shape(moments["cov"])[0]
        tril_mask = self.make_tril_bool_mask(batch_size)

        # Compute reactions
        params0TEforRxns = []
        for i in range(0,len(self.rxns)):
            # print("--- Rxn idx: %d ---" % i)

            momentsTE = self.rxns[i](moments)

            # Convert nMomentsTE to params0TE
            momentsTE.update(moments)
            params0TE = self.momentsTEtoParams0TE(momentsTE)

            # Flatten
            # Reshape (batch_size, a, b) into (batch_size, a*b) for each thing in the dict
            cholv_TE_std_vec = tf.boolean_mask(params0TE["cholv_TE_std"], tril_mask)
            params0TEarr = [
                tf.reshape(cholv_TE_std_vec, (batch_size, int(self.nv*(self.nv+1)/2))),
                tf.reshape(params0TE["muv_TE"], (batch_size, self.nv))
                ]

            # Combine different tensors of size (batch_size, a), (batch_size, b), ... 
            # into one of (batch_size, a+b+...)
            params0TEflat = tf.concat(params0TEarr, 1)

            # Store
            params0TEforRxns.append(params0TEflat)

        # Flatten all reactions
        # Combine different tensors of size (batch_size, a), (batch_size, b), ... 
        # into one of (batch_size, a+b+...)
        params0TE = tf.concat(params0TEforRxns, 1)

        return params0TE