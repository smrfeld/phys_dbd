import tensorflow as tf

import numpy as np
from typing import List, Tuple, Union, Any, Dict

from enum import Enum

from tensorflow.python.ops.gen_resource_variable_ops import var_handle_op

# Make new layers and models via subclassing
# https://www.tensorflow.org/guide/keras/custom_layers_and_models

# tf.keras.utils.register_keras_serializable(package="Custom", name=None)
# This decorator injects the decorated class or function into the Keras 
# custom object dictionary, so that it can be serialized and deserialized 
# without needing an entry in the user-provided custom object dict. It also 
# injects a function that Keras will call to get the object's 
# serializable string key.
# https://keras.io/api/utils/serialization_utils/

@tf.keras.utils.register_keras_serializable(package="physDBD")
class FourierLatentLayer(tf.keras.layers.Layer):

    def __init__(self, 
        freqs : np.array,
        offset_fixed : float,
        sin_coeff : np.array,
        cos_coeff : np.array,
        **kwargs
        ):
        """Fourier latent layer

        Args:
            freqs (np.array): 1D arr of frequencies of length L
            offset_fixed (float): Float offset (not trained)
            sin_coeff (np.array): 1D arr of sin coeffs (trained) of length L
            cos_coeff (np.array): 1D arr of sin coeffs (trained) of length L
        """
        # Super
        super(FourierLatentLayer, self).__init__(**kwargs)

        self.freqs = self.add_weight(
            name="freqs",
            shape=len(freqs),
            trainable=False,
            initializer=tf.constant_initializer(freqs),
            dtype='float32'
            )
        
        # Add weight does the same as tf.Variable + initializer
        # Note you also have access to a quicker shortcut for 
        # adding weight to a layer: the add_weight() method:

        self.offset_fixed = self.add_weight(
            name="offset_fixed",
            shape=1,
            trainable=False,
            initializer=tf.constant_initializer(offset_fixed),
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

    # https://keras.io/guides/serialization_and_saving/#custom-objects
    def get_config(self):
        config = super(FourierLatentLayer, self).get_config()
        config.update({
            "freqs": self.freqs.numpy(),
            "offset_fixed": self.offset_fixed.numpy(),
            "cos_coeff": self.cos_coeff.numpy(),
            "sin_coeff": self.sin_coeff.numpy()
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Build function is only needed if the input is not known until runtime
    # The __call__() method of your layer will automatically run build the first 
    # time it is called. You now have a layer that's lazy and thus easier to use:
    # input_shape is the input in the call method
    # def build(self, input_shape):
    
    # @tf.function
    def fourier_range(self):
        """Compute the denominator = max(1, sum(abs(coeffs)))

        Returns:
            float: denominator
        """
        st = tf.math.reduce_sum(abs(self.sin_coeff))
        ct = tf.math.reduce_sum(abs(self.cos_coeff))
        return tf.math.maximum(1.0, st+ct+1.e-8)

    def call(self, inputs):
        
        # tf.math.reduce_sum is same as np.sum
        # tf.matmul is same np.dot
        tsin = tf.map_fn(lambda tpt: tf.math.sin(tpt * self.freqs), inputs["tpt"])

        # Dot product = tf.tensordot(a, b, 1)
        ts = tf.map_fn(lambda tsinL: tf.tensordot(self.sin_coeff, tsinL, 1), tsin)

        # Same for cos
        tcos = tf.map_fn(lambda tpt: tf.math.cos(tpt * self.freqs), inputs["tpt"])
        tc = tf.map_fn(lambda tcosL: tf.tensordot(self.cos_coeff, tcosL, 1), tcos)

        return self.offset_fixed + (ts + tc) / self.fourier_range()

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParamsLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """Convert params latent space to different muh,varh_diag
        """
        # Super
        super(ConvertParamsLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super(ConvertParamsLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs):

        b1 = inputs["b1"]
        wt1 = inputs["wt1"]
        muh1 = inputs["muh1"]
        muh2 = inputs["muh2"]
        varh_diag1 = inputs["varh_diag1"]
        varh_diag2 = inputs["varh_diag2"]

        # Dim 0 = batch size
        # To take the transpose of the matrices in dimension-0 (such as when you are transposing matrices 
        # where 0 is the batch dimension), you would set perm=[0,2,1].
        # https://www.tensorflow.org/api_docs/python/tf/transpose
        w1 = tf.transpose(wt1, perm=[0,2,1])

        varh1_sqrt = tf.map_fn(
            lambda varh_diagL: tf.math.sqrt(tf.linalg.tensor_diag(varh_diagL)), 
            varh_diag1)
        varh2_inv_sqrt = tf.map_fn(
            lambda varh_diagL: tf.linalg.tensor_diag(tf.math.sqrt(tf.math.pow(varh_diagL,-1))), 
            varh_diag2)

        # Matrix * vector = tf.linalg.matvec
        # Diagonal matrix = tf.linalg.tensor_diag
        
        # Matmul batch:
        # The inputs must, following any transpositions, be tensors of rank >= 2 where the inner 2 dimensions specify 
        # valid matrix multiplication dimensions, and any further outer dimensions specify matching batch size.
        # https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
        # i.e. matmul already does batch multiplication correctly
        # same for matvec
        m2 = tf.matmul(w1, tf.matmul(varh1_sqrt, varh2_inv_sqrt))
        b2 = b1 + tf.linalg.matvec(w1, muh1) - tf.linalg.matvec(m2, muh2)

        wt2 = tf.matmul(varh2_inv_sqrt, tf.matmul(varh1_sqrt, wt1))

        return {
            "b2": b2,
            "wt2": wt2
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParamsLayerFrom0(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """Convert params0 = std. params with muh=0, varh_diag=1 to params with different muh,varh_diag
        """
        # Super
        super(ConvertParamsLayerFrom0, self).__init__(**kwargs)

    def get_config(self):
        config = super(ConvertParamsLayerFrom0, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs):
        b1 = inputs["b1"]
        wt1 = inputs["wt1"]
        muh2 = inputs["muh2"]
        varh_diag2 = inputs["varh_diag2"]

        w1 = tf.transpose(wt1,perm=[0,2,1])

        varh2_inv_sqrt = tf.map_fn(
            lambda varh_diagL: tf.linalg.tensor_diag(tf.math.sqrt(tf.math.pow(varh_diagL,-1))),
            varh_diag2)

        # Matrix * vector = tf.linalg.matvec
        # Diagonal matrix = tf.linalg.tensor_diag
        m2 = tf.matmul(w1, varh2_inv_sqrt)
        b2 = b1 - tf.linalg.matvec(m2, muh2)

        wt2 = tf.matmul(varh2_inv_sqrt, wt1)

        return {
            "b2": b2,
            "wt2": wt2
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParams0ToParamsLayer(tf.keras.layers.Layer):

    def __init__(self, 
        nv: int, 
        nh: int, 
        layer_muh: Dict[str,FourierLatentLayer],
        layer_varh_diag: Dict[str,FourierLatentLayer],
        **kwargs):
        """Convert params0 to params layer using Fourier to represent latents.
           Params0 = std. params with muh=0,varh_diag=1
           Output = params with different muh,varh_diag obtained from the Fourier representation


        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
            layer_muh (Dict[str,FourierLatentLayer]): Keys = "0", "1", ..., "nh". Note that TF only likes string keys. Values = FourierLatentLayer.
            layer_varh_diag (Dict[str,FourierLatentLayer]): Keys = "0", "1", ..., "nh". Note that TF only likes string keys. Values = FourierLatentLayer.
        """

        # Super
        super(ConvertParams0ToParamsLayer, self).__init__(**kwargs)

        self.nv = nv
        self.nh = nh

        self.layer_muh = layer_muh
        self.layer_varh_diag = layer_varh_diag

        self.convert_from_0 = ConvertParamsLayerFrom0()

    @classmethod
    def construct(cls, 
        nv : int,
        nh : int,
        freqs : np.array,
        muh_sin_coeffs_init : np.array,
        muh_cos_coeffs_init : np.array,
        varh_sin_coeffs_init : np.array,
        varh_cos_coeffs_init : np.array,
        **kwargs):
        """Construct the layer including the FourierLatentLayers from initial coefficients.

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
            freqs (np.array): 1D arr of frequencies of length L
            muh_sin_coeffs_init (np.array): 1D arr of initial coeffs of length L
            muh_cos_coeffs_init (np.array): 1D arr of initial coeffs of length L
            varh_sin_coeffs_init (np.array): 1D arr of initial coeffs of length L
            varh_cos_coeffs_init (np.array): 1D arr of initial coeffs of length L
        """

        layer_muh = {}
        layer_varh_diag = {}
        for ih in range(0,nh):

            # Only use strings as keys
            # https://stackoverflow.com/a/57974800/1427316
            layer_muh[str(ih)] = FourierLatentLayer(
                freqs=freqs,
                offset_fixed=0.0,
                sin_coeff=muh_sin_coeffs_init,
                cos_coeff=muh_cos_coeffs_init
                )

            layer_varh_diag[str(ih)] = FourierLatentLayer(
                freqs=freqs,
                offset_fixed=1.01,
                sin_coeff=varh_sin_coeffs_init,
                cos_coeff=varh_cos_coeffs_init
                )

        return cls(
            nv=nv,
            nh=nh,
            layer_muh=layer_muh,
            layer_varh_diag=layer_varh_diag,
            **kwargs
            )

    def get_config(self):
        config = super(ConvertParams0ToParamsLayer, self).get_config()

        config.update({
            "nv": self.nv,
            "nh": self.nh,
            "layer_muh": self.layer_muh,
            "layer_varh_diag": self.layer_varh_diag
        })
        
        return config
    
    @classmethod
    def from_config(cls, config):

        layer_muh = {}
        for key, val in config["layer_muh"].items():
            layer_muh[key] = FourierLatentLayer(**val['config'])
        
        layer_varh_diag = {}
        for key, val in config["layer_varh_diag"].items():
            layer_varh_diag[key] = FourierLatentLayer(**val['config'])

        config["layer_muh"] = layer_muh
        config["layer_varh_diag"] = layer_varh_diag

        return cls(**config)
    
    def call(self, inputs):

        # Get fourier
        muhs = []
        varh_diags = []
        for ih in range(0,self.nh):
            muhs.append(self.layer_muh[str(ih)](inputs))
            varh_diags.append(self.layer_varh_diag[str(ih)](inputs))
        
        # Current size is (nh, batch_size)
        # Transpose to get (batch_size, nh)
        muh = tf.transpose(muhs)
        varh_diag = tf.transpose(varh_diags)
        # muh = tf.concat(muhs,0)
        # varh_diag = tf.concat(varh_diags,0)

        inputs_convert = {
            "muh2": muh,
            "varh_diag2": varh_diag,
            "b1": inputs["b"],
            "wt1": inputs["wt"]
        }
        outputs_convert = self.convert_from_0(inputs_convert)

        output = {
            "sig2": inputs["sig2"],
            "wt": outputs_convert["wt2"],
            "b": outputs_convert["b2"],
            "muh": muh,
            "varh_diag": varh_diag
        }

        return output

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParamsToMomentsLayer(tf.keras.layers.Layer):

    def __init__(self,
        nv : int,
        nh : int,
        **kwargs
        ):
        """Convert params to moments.
           Params = (wt,b,sig2,muh,varh_diag)
           Moments = (mean, cov_matrix)

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
        """
        # Super
        super(ConvertParamsToMomentsLayer, self).__init__(**kwargs)
        self.nv = nv 
        self.nh = nh
    
    def get_config(self):
        config = super(ConvertParamsToMomentsLayer, self).get_config()
        config.update({
            "nv": self.nv,
            "nh": self.nh
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):

        wt = inputs["wt"]
        varh_diag = inputs["varh_diag"]
        sig2 = inputs["sig2"]
        b = inputs["b"]
        muh = inputs["muh"]

        w = tf.transpose(wt, perm=[0,2,1])
        varh = tf.map_fn(lambda varh_diagL: tf.linalg.tensor_diag(varh_diagL), varh_diag)

        varvh = tf.matmul(varh, wt)
        varvTMP = tf.matmul(w,tf.matmul(varh,wt))
        sig2TMP = tf.map_fn(lambda sig2L: sig2L * tf.eye(self.nv), sig2)
        varv = varvTMP + sig2TMP

        muv = b + tf.linalg.matvec(w, muh)

        # Assemble mu
        mu = tf.concat([muv,muh],1)

        # Array flatten vars
        varvht = tf.transpose(varvh, perm=[0,2,1])
        var1 = tf.concat([varv,varvht],2)
        var2 = tf.concat([varvh,varh],2)
        var = tf.concat([var1,var2],1)

        return {
            "mu": mu,
            "var": var
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertMomentsToNMomentsLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """Convert moments to nMoments
           Moments = (mean, cov_mat)
           nMoments = (mean, n^2 matrix = cov_mat + mean.mean^T)
        """
        # Super
        super(ConvertMomentsToNMomentsLayer, self).__init__(**kwargs)
    
    def get_config(self):
        config = super(ConvertMomentsToNMomentsLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        
        mu = inputs["mu"]
        var = inputs["var"]

        # kronecker product of two vectors = tf.tensordot(a,b,axes=0)
        kpv = tf.map_fn(lambda muL: tf.tensordot(muL,muL,axes=0),mu)

        nvar = var + kpv

        return {
            "mu": mu,
            "nvar": nvar
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParams0ToNMomentsLayer(tf.keras.layers.Layer):

    def __init__(self, 
        nv: int,
        nh: int,
        params0ToParamsLayer: ConvertParams0ToParamsLayer,
        **kwargs
        ):
        """Convert params0 to NMoments layer in one step
           Params0 = std. params with muh=0,varh_diag=1
           nMoments = (mean, n^2 matrix = cov_mat + mean.mean^T)

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
            params0ToParamsLayer (ConvertParams0ToParamsLayer): Conversion layer from params0 to params via latent Fourier representation
        """
        # Super
        super(ConvertParams0ToNMomentsLayer, self).__init__(**kwargs)

        self.nv = nv
        self.nh = nh

        self.params0ToParamsLayer = params0ToParamsLayer
        self.paramsToMomentsLayer = ConvertParamsToMomentsLayer(nv,nh)
        self.momentsToNMomentsLayer = ConvertMomentsToNMomentsLayer()

    @classmethod
    def construct(cls,
        nv: int, 
        nh: int,
        freqs : np.array,
        muh_sin_coeffs_init : np.array,
        muh_cos_coeffs_init : np.array,
        varh_sin_coeffs_init : np.array,
        varh_cos_coeffs_init : np.array,
        **kwargs
        ):
        """Construct the layer including the Fourier latent representations.

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
            freqs (np.array): 1D arr of frequencies of length L
            muh_sin_coeffs_init (np.array): 1D arr of coefficients of length L
            muh_cos_coeffs_init (np.array): 1D arr of coefficients of length L
            varh_sin_coeffs_init (np.array): 1D arr of coefficients of length L
            varh_cos_coeffs_init (np.array): 1D arr of coefficients of length L
        """
        params0ToParamsLayer = ConvertParams0ToParamsLayer.construct(
            nv=nv, 
            nh=nh, 
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            varh_sin_coeffs_init=varh_sin_coeffs_init,
            varh_cos_coeffs_init=varh_cos_coeffs_init
            )

        return cls(
            nv=nv,
            nh=nh,
            params0ToParamsLayer=params0ToParamsLayer,
            **kwargs
            )

    def get_config(self):
        config = super(ConvertParams0ToNMomentsLayer, self).get_config()
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

        dall = {}
        for d in [params,moments,nmoments]:
            dall.update(d)
        
        return dall

# @tf.function
@tf.keras.utils.register_keras_serializable(package="physDBD")
def unit_mat_sym(n: int, i: int, j: int):
    """Construct the symmetric unit matrix of size nxn
       1 at (i,j) AND (j,i)
       0 elsewhere

    Args:
        n (int): Size of square matrix
        i (int): First idx
        j (int): Second idx

    Returns:
        tf.Constant: Matrix that is 1 at (i,j) AND (j,i) and 0 everywhere else
    """
    idx = i * n + j
    one_hot = tf.one_hot(indices=idx,depth=n*n, dtype='float32')
    
    if i != j:
        idx = j * n + i
        one_hot += tf.one_hot(indices=idx,depth=n*n, dtype='float32')

    return tf.reshape(one_hot,shape=(n,n))

@tf.keras.utils.register_keras_serializable(package="physDBD")
class DeathRxnLayer(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, i_sp: int, **kwargs):
        """Death reaction A->0 acting on nMoments = (mean, n^2 matrix = cov_mat + mean.mean^T)

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
            i_sp (int): Species A->0 that is decaying
        """
        # Super
        super(DeathRxnLayer, self).__init__(**kwargs)
    
        self.nv = nv
        self.nh = nh
        self.i_sp = i_sp

        self.n = nv + nh

    def get_config(self):
        config = super(DeathRxnLayer, self).get_config()
        config.update({
            "nv": self.nv,
            "nh": self.nh,
            "i_sp": self.i_sp
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs):

        mu = inputs["mu"]
        nvar = inputs["nvar"]

        unit = tf.one_hot(
            indices=self.i_sp,
            depth=self.n
            )

        muTE = tf.map_fn(lambda muL: - muL[self.i_sp] * unit, mu)
        
        # tf.zeros_like vs tf.zeros
        # https://stackoverflow.com/a/49599952/1427316
        # avoid 'Cannot convert a partially known TensorShape to a Tensor'
        nvarTE = tf.zeros_like(nvar)
        for j in range(0,self.n):
            if j == self.i_sp:
                vals = -2.0 * nvar[:,self.i_sp,self.i_sp] + mu[:,self.i_sp]
            else:
                vals = -1.0 * nvar[:,self.i_sp,j]

            unit_mat = unit_mat_sym(self.n,self.i_sp,j)
            nvarTE += tf.map_fn(lambda val: unit_mat * val, vals)
        
        return {
            "muTE": muTE,
            "nvarTE": nvarTE
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class BirthRxnLayer(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, i_sp: int, **kwargs):
        """Birth reaction A->2A acting on nMoments = (mean, n^2 matrix = cov_mat + mean.mean^T)

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
            i_sp (int): Species A->2A that is reproducing
        """
        # Super
        super(BirthRxnLayer, self).__init__(**kwargs)
    
        self.nv = nv
        self.nh = nh
        self.i_sp = i_sp

        self.n = nv + nh

    def get_config(self):
        config = super(BirthRxnLayer, self).get_config()
        config.update({
            "nv": self.nv,
            "nh": self.nh,
            "i_sp": self.i_sp
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs):
        
        mu = inputs["mu"]
        nvar = inputs["nvar"]

        unit = tf.one_hot(
            indices=self.i_sp,
            depth=self.n
            )

        muTE = tf.map_fn(lambda muL: muL[self.i_sp] * unit, mu)

        nvarTE = tf.zeros_like(nvar)
        for j in range(0,self.n):
            if j == self.i_sp:
                vals = 2.0 * nvar[:,self.i_sp,self.i_sp] + mu[:,self.i_sp]
            else:
                vals = nvar[:,self.i_sp,j]

            unit_mat = unit_mat_sym(self.n,self.i_sp,j)
            nvarTE += tf.map_fn(lambda val: unit_mat * val, vals)
        
        return {
            "muTE": muTE,
            "nvarTE": nvarTE
        }

# @tf.function
@tf.keras.utils.register_keras_serializable(package="physDBD")
def nmoment3_batch(mu, nvar, i, j, k):
    return -2.0*mu[:,i]*mu[:,j]*mu[:,k] + mu[:,i]*nvar[:,j,k] + mu[:,j]*nvar[:,i,k] + mu[:,k]*nvar[:,i,j]

@tf.keras.utils.register_keras_serializable(package="physDBD")
class EatRxnLayer(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, i_hunter: int, i_prey: int, **kwargs):
        """Predator-prey reaction H+P->2H acting on nMoments = (mean, n^2 matrix = cov_mat + mean.mean^T)

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
            i_hunter (int): Hunter species
            i_prey (int): Prey species
        """
        # Super
        super(EatRxnLayer, self).__init__(**kwargs)
    
        self.nv = nv
        self.nh = nh
        self.i_hunter = i_hunter
        self.i_prey = i_prey

        self.n = nv + nh

    def get_config(self):
        config = super(EatRxnLayer, self).get_config()
        config.update({
            "nv": self.nv,
            "nh": self.nh,
            "i_hunter": self.i_hunter,
            "i_prey": self.i_prey
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    # @tf.function
    def nvar_prey_prey(self, mu, nvar):
        n3 = nmoment3_batch(mu,nvar,self.i_prey,self.i_prey,self.i_hunter)
        vals = -2.0*n3 + nvar[:, self.i_hunter, self.i_prey]
        unit_mat = unit_mat_sym(self.n, self.i_prey, self.i_prey)
        return tf.map_fn(lambda val: unit_mat * val, vals)

    # @tf.function
    def nvar_hunter_hunter(self, mu, nvar):
        n3 = nmoment3_batch(mu,nvar,self.i_hunter,self.i_hunter,self.i_prey)
        vals = 2.0*n3 + nvar[:,self.i_hunter, self.i_prey]
        unit_mat = unit_mat_sym(self.n, self.i_hunter, self.i_hunter)
        return tf.map_fn(lambda val: unit_mat * val, vals)

    # @tf.function
    def nvar_hunter_prey(self, mu, nvar):
        n3hhp = nmoment3_batch(mu,nvar,self.i_hunter,self.i_hunter,self.i_prey)
        n3hpp = nmoment3_batch(mu,nvar,self.i_hunter,self.i_prey,self.i_prey)
        unit_mat = unit_mat_sym(self.n, self.i_hunter, self.i_prey)
        vals = - n3hhp + n3hpp - nvar[:,self.i_hunter, self.i_prey] 
        return tf.map_fn(lambda val: unit_mat * val, vals)

    # @tf.function
    def nvar_loop(self, mu, nvar, j : int):
        um_prey = unit_mat_sym(self.n, self.i_prey, j)
        um_hunter = unit_mat_sym(self.n, self.i_hunter, j)
        unit_mat = um_hunter - um_prey
        n3 = nmoment3_batch(mu,nvar,j,self.i_prey,self.i_hunter)
        vals = n3
        return tf.map_fn(lambda val: unit_mat * val, vals)

    def call(self, inputs):
        
        mu = inputs["mu"]
        nvar = inputs["nvar"]

        unit_hunter = tf.one_hot(
            indices=self.i_hunter,
            depth=self.n
            )
        unit_prey = tf.one_hot(
            indices=self.i_prey,
            depth=self.n
            )
        
        muTE = tf.map_fn(
            lambda nvarL: - nvarL[self.i_hunter, self.i_prey] * unit_prey \
                + nvarL[self.i_hunter,self.i_prey] * unit_hunter,
                nvar)
        
        nvarTE = tf.zeros_like(nvar)

        # Prey-prey
        nvarTE += self.nvar_prey_prey(mu,nvar)
        
        # Hunter-hunter
        nvarTE += self.nvar_hunter_hunter(mu,nvar)

        # Hunter-prey
        nvarTE += self.nvar_hunter_prey(mu,nvar)

        # Loop
        for j in range(0,self.n):
            if j != self.i_prey and j != self.i_hunter:
                nvarTE += self.nvar_loop(mu,nvar,j)

        return {
            "muTE": muTE,
            "nvarTE": nvarTE
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertNMomentsTEtoMomentsTE(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """Convert nMomentsTE = nMoments time evolution to momentsTE = moments time evolution
            nMomentsTE = time evolution of (mean, n^2 matrix = cov_mat + mean.mean^T)
            momentsTE = time evolution of (mean, cov_mat)
        """
        # Super
        super(ConvertNMomentsTEtoMomentsTE, self).__init__(**kwargs)

    def get_config(self):
        config = super(ConvertNMomentsTEtoMomentsTE, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        
        mu = inputs["mu"]
        muTE = inputs["muTE"]
        nvarTE = inputs["nvarTE"]

        # kronecker product of two vectors = tf.tensordot(a,b,axes=0)
        # neg_kpv = - tf.tensordot(muTE,mu,axes=0) - tf.tensordot(mu,muTE,axes=0)

        # For batch mode:
        # Just use tf.multiply or equivalently * operator
        # (batch, n, 1) * (batch, 1, n) gives (batch, n, n) 
        # https://stackoverflow.com/a/51641382/1427316

        # Do not use .shape; rather use tf.shape
        # However, h = tf.shape(x)[1]; w = tf.shape(x)[2] will let h, w be 
        # symbolic or graph-mode tensors (integer) that will contain a dimension 
        # of x. The value will be determined runtime. In such a case, tf.reshape(x, [-1, w, h]) will 
        # produce a (symbolic) tensor of shape [?, ?, ?] (still unknown) whose tensor shape 
        # will be known on runtime.
        # https://github.com/tensorflow/models/issues/6245#issuecomment-623877638
        batch_size = tf.shape(mu)[0]
        n = tf.shape(mu)[1]

        muTE1 = tf.reshape(muTE,shape=(batch_size,n,1))
        mu1 = tf.reshape(mu,shape=(batch_size,1,n))

        mu2 = tf.reshape(mu,shape=(batch_size,n,1))
        muTE2 = tf.reshape(muTE,shape=(batch_size,1,n))

        neg_kpv = - tf.multiply(muTE1,mu1) - tf.matmul(mu2,muTE2)

        varTE = nvarTE + neg_kpv

        return {
            "muTE": muTE,
            "varTE": varTE
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertMomentsTEtoParamMomentsTE(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, **kwargs):
        """Convert momentsTE = moments time evolution to paramMomentsTE = param moments time evolution
            momentsTE = time evolution of (mean, cov_mat)
            paramMomentsTE = time evolution of (
                    mean visible species, 
                    varvh = off diagonal matrix in cov mat, 
                    trace of diagonal of visible part of cov matrix, 
                    muh = mean of hidden species, 
                    varh_diag = diagonal of hidden part of cov matrix
                    ) 
                corresponding in dimensionality to (b,wt,sig2,muh,varh_diag)
        """
        # Super
        super(ConvertMomentsTEtoParamMomentsTE, self).__init__(**kwargs)
    
        self.nv = nv
        self.nh = nh

    def get_config(self):
        config = super(ConvertMomentsTEtoParamMomentsTE, self).get_config()
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
        varTE = inputs["varTE"]

        muvTE = muTE[:,:self.nv]
        muhTE = muTE[:,self.nv:]
        varvhTE = varTE[:,self.nv:,:self.nv]
        varvbarTE = tf.map_fn(
            lambda varTEL: tf.linalg.trace(varTEL[:self.nv,:self.nv]), 
            varTE)
        varh_diagTE = tf.map_fn(
            lambda varTEL: tf.linalg.diag_part(varTEL[self.nv:,self.nv:]),
            varTE)

        return {
            "muvTE": muvTE,
            "muhTE": muhTE,
            "varvhTE": varvhTE,
            "varvbarTE": varvbarTE,
            "varh_diagTE": varh_diagTE
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParamMomentsTEtoParamsTE(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, **kwargs):
        """Convert time evolution of param moments to time evolution of params
            paramMomentsTE = time evolution of (
                    mean visible species, 
                    varvh = off diagonal matrix in cov mat, 
                    trace of diagonal of visible part of cov matrix, 
                    muh = mean of hidden species, 
                    varh_diag = diagonal of hidden part of cov matrix
                    ) 
                corresponding in dimensionality to (b,wt,sig2,muh,varh_diag)
            paramsTE = time evolution of (b,wt,sig2,muh,varh_diag)

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
        """
        # Super
        super(ConvertParamMomentsTEtoParamsTE, self).__init__(**kwargs)
    
        self.nv = nv
        self.nh = nh

    def get_config(self):
        config = super(ConvertParamMomentsTEtoParamsTE, self).get_config()
        config.update({
            "nv": self.nv,
            "nh": self.nh
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):

        muvTE = inputs["muvTE"]
        varvhTE = inputs["varvhTE"]
        varh_diagTE = inputs["varh_diagTE"]
        varh_diag = inputs["varh_diag"]
        muh = inputs["muh"]
        varvh = inputs["varvh"]
        muhTE = inputs["muhTE"]
        varvbarTE = inputs["varvbarTE"]

        varh_inv = tf.map_fn(
            lambda varh_diagL: tf.linalg.tensor_diag(1.0 / varh_diagL), 
            varh_diag)
        varhTE = tf.map_fn(
            lambda varh_diagTEL: tf.linalg.tensor_diag(varh_diagTEL),
            varh_diagTE)

        varvh_Trans = tf.transpose(varvh,perm=[0,2,1])
        varvhTE_Trans = tf.transpose(varvhTE,perm=[0,2,1])

        bTE = muvTE 
        bTE -= tf.linalg.matvec(varvhTE_Trans, 
            tf.linalg.matvec(varh_inv, muh))
        bTE += tf.linalg.matvec(varvh_Trans, 
            tf.linalg.matvec(varh_inv, 
            tf.linalg.matvec(varhTE, 
            tf.linalg.matvec(varh_inv, muh))))
        bTE -= tf.linalg.matvec(varvh_Trans,
            tf.linalg.matvec(varh_inv, muhTE))
        
        wtTE = - tf.matmul(varh_inv,
            tf.matmul(varhTE,
            tf.matmul(varh_inv,varvh)))
        wtTE += tf.matmul(varh_inv, varvhTE)

        sig2TEmat = tf.matmul(varvhTE_Trans,
            tf.matmul(varh_inv, varvh))
        sig2TEmat -= tf.matmul(varvh_Trans,
            tf.matmul(varh_inv,
            tf.matmul(varhTE,
            tf.matmul(varh_inv,varvh))))
        sig2TEmat += tf.matmul(varvh_Trans,
            tf.matmul(varh_inv,varvhTE))
        sig2TEtr = tf.map_fn(
            lambda sig2TEmatL: tf.linalg.trace(sig2TEmatL), 
            sig2TEmat)
        sig2TE = (varvbarTE - sig2TEtr) / self.nv

        return {
            "bTE": bTE,
            "wtTE": wtTE,
            "sig2TE": sig2TE,
            "muhTE": muhTE,
            "varh_diagTE": varh_diagTE
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParamsTEtoParams0TE(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """Convert paramsTE = time evolution of params to params0TE = time evolution of params0
            paramsTE = time evolution of (b,wt,sig2,muh,varh_diag)
            params0TE = time evolution of (b,wt,sig2) in std. space with muh=0, varh_diag=1, 
                both constant in time i.e. muhTE=0, varh_diagTE=0
        """
        # Super
        super(ConvertParamsTEtoParams0TE, self).__init__(**kwargs)

    def get_config(self):
        config = super(ConvertParamsTEtoParams0TE, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs):

        bTE1 = inputs["bTE1"]
        wtTE1 = inputs["wtTE1"]
        muh1 = inputs["muh1"]
        wt1 = inputs["wt1"]
        muhTE1 = inputs["muhTE1"]
        varh_diag1 = inputs["varh_diag1"]
        varh_diagTE1 = inputs["varh_diagTE1"]
        sig2TE = inputs["sig2TE"]

        varh1 = tf.map_fn(lambda varh_diag1L: tf.linalg.tensor_diag(varh_diag1L), varh_diag1)
        varh_inv1 = tf.map_fn(lambda varh_diag1L: tf.linalg.tensor_diag(1.0 / varh_diag1L), varh_diag1)
        varhTE1 = tf.map_fn(lambda varh_diagTE1L: tf.linalg.tensor_diag(varh_diagTE1L), varh_diagTE1)

        wTE1 = tf.transpose(wtTE1,perm=[0,2,1])
        w1 = tf.transpose(wt1,perm=[0,2,1])

        bTE2 = bTE1 + tf.linalg.matvec(wTE1,muh1) + tf.linalg.matvec(w1,muhTE1)
        wtTE2 = 0.5 * tf.matmul(tf.math.sqrt(varh_inv1),
            tf.matmul(varhTE1, wt1)) + tf.matmul(tf.math.sqrt(varh1),wtTE1)
        
        return {
            "sig2TE": sig2TE,
            "bTE2" : bTE2,
            "wtTE2" : wtTE2
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertNMomentsTEtoParams0TE(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, **kwargs):
        """Convert nMomentsTE = time evolution of nMoments to params0TE = time evolution of params0 in one step
            nMomentsTE = time evolution of (mean, n^2 matrix = cov_mat + mean.mean^T)
            params0TE = time evolution of (b,wt,sig2) in std. space with muh=0, varh_diag=1, 
                both constant in time i.e. muhTE=0, varh_diagTE=0

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
        """

        # Super
        super(ConvertNMomentsTEtoParams0TE, self).__init__(**kwargs)
        
        self.nv = nv
        self.nh = nh

        self.nMomentsTEtoMomentsTE = ConvertNMomentsTEtoMomentsTE()
        self.momentsTEtoParamMomentsTE = ConvertMomentsTEtoParamMomentsTE(nv,nh)
        self.paramMomentsTEtoParamsTE = ConvertParamMomentsTEtoParamsTE(nv,nh)
        self.paramsTEtoParams0TE = ConvertParamsTEtoParams0TE()

    def get_config(self):
        config = super(ConvertNMomentsTEtoParams0TE, self).get_config()
        config.update({
            "nv": self.nv,
            "nh": self.nh
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        outputs1 = self.nMomentsTEtoMomentsTE(inputs)

        outputs2 = self.momentsTEtoParamMomentsTE(outputs1)

        outputs2["varh_diag"] = inputs["varh_diag"]
        outputs2["muh"] = inputs["muh"]
        outputs2["varvh"] = inputs["var"][:,self.nv:,:self.nv]
        outputs3 = self.paramMomentsTEtoParamsTE(outputs2)

        inputs4 = {
            "bTE1": outputs3["bTE"],
            "wtTE1": outputs3["wtTE"],
            "muh1": inputs["muh"],
            "wt1": inputs["wt"],
            "muhTE1": outputs3["muhTE"],
            "varh_diag1": inputs["varh_diag"],
            "varh_diagTE1": outputs3["varh_diagTE"],
            "sig2TE": outputs3["sig2TE"]
        }
        outputs4 = self.paramsTEtoParams0TE(inputs4)

        return {
            "sig2TE": outputs4["sig2TE"],
            "bTE": outputs4["bTE2"],
            "wtTE": outputs4["wtTE2"]
        }

# Model vs Layer
# https://www.tensorflow.org/tutorials/customization/custom_layers#models_composing_layers
# Typically you inherit from keras.Model when you need the model methods like: Model.fit,Model.evaluate, 
# and Model.save (see Custom Keras layers and models for details).
# One other feature provided by keras.Model (instead of keras.layers.Layer) is that in addition to 
# tracking variables, a keras.Model also tracks its internal layers, making them easier to inspect.
@tf.keras.utils.register_keras_serializable(package="physDBD")
class RxnInputsLayer(tf.keras.layers.Layer):

    def __init__(self, 
        nv: int, 
        nh: int, 
        params0toNMoments: ConvertParams0ToNMomentsLayer,
        rxn_specs : List[Union[Tuple[str,int],Tuple[str,int,int]]],
        **kwargs
        ):
        """Calculate inputs from different reaction approximations in a single layer. 
        Input is params0 = (wt,b,sig2) in std. space where muh=0, varh_diag=1
        Output is for every reaction in the rxn specs, the time evolution of the params0 = params0TE = (wtTE, bTE, sig2TE), flattened across all reactions

        Args:
            nv (int): No. visible species
            nh (int): No. hidden species
            params0toNMoments (ConvertParams0ToNMomentsLayer): Layer that converts params0 to nMoments
            rxn_specs (List[Union[Tuple[str,int],Tuple[str,int,int]]]): List of reaction specifications of tuples.
                First arguments is one of "BIRTH", "DEATH" or "EAT"
                Second/third argument are the index of the species undergoing the reaction
                e.g. ("EAT",3,2) is the Predator-Prey reaction for species #3 being the huner, species #2 the prey

        Raises:
            ValueError: If reaaction string is not recognized
        """
        # Super
        super(RxnInputsLayer, self).__init__(**kwargs)
        
        self.nv = nv
        self.nh = nh

        self.params0toNMoments = params0toNMoments
        self.nMomentsTEtoParams0TE = ConvertNMomentsTEtoParams0TE(nv,nh)

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
    def construct(cls, 
        nv: int, 
        nh: int, 
        freqs : np.array,
        muh_sin_coeffs_init : np.array,
        muh_cos_coeffs_init : np.array,
        varh_sin_coeffs_init : np.array,
        varh_cos_coeffs_init : np.array,
        rxn_specs : List[Union[Tuple[str,int],Tuple[str,int,int]]],
        **kwargs
        ):
        
        params0toNMoments = ConvertParams0ToNMomentsLayer.construct(
            nv=nv,
            nh=nh,
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            varh_sin_coeffs_init=varh_sin_coeffs_init,
            varh_cos_coeffs_init=varh_cos_coeffs_init
            )

        return cls(
            nv=nv,
            nh=nh,
            params0toNMoments=params0toNMoments,
            rxn_specs=rxn_specs,
            **kwargs
            )

    def get_config(self):
        config = super(RxnInputsLayer, self).get_config()

        config.update({
            "nv": self.nv,
            "nh": self.nh,
            "params0toNMoments": self.params0toNMoments,
            "rxn_specs": self.rxn_specs
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        nMoments = self.params0toNMoments(inputs)
        batch_size = tf.shape(nMoments["nvar"])[0]

        # Compute reactions
        params0TEforRxns = []
        for i in range(0,len(self.rxns)):
            # print("--- Rxn idx: %d ---" % i)

            nMomentsTE = self.rxns[i](nMoments)

            # Convert nMomentsTE to params0TE
            nMomentsTE.update(nMoments)
            params0TE = self.nMomentsTEtoParams0TE(nMomentsTE)

            # Flatten
            # Reshape (batch_size, a, b) into (batch_size, a*b) for each thing in the dict
            params0TEarr = [
                tf.reshape(params0TE["wtTE"], (batch_size, self.nh * self.nv)),
                tf.reshape(params0TE["bTE"], (batch_size, self.nv)),
                tf.reshape(params0TE["sig2TE"], (batch_size,1))
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