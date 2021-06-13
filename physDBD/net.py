import tensorflow as tf

import numpy as np
from typing import List, Tuple, Union

from enum import Enum

# Make new layers and models via subclassing
# https://www.tensorflow.org/guide/keras/custom_layers_and_models

class FourierLatentLayer(tf.keras.layers.Layer):

    def __init__(self, 
        freqs : np.array,
        offset_fixed : float,
        sin_coeffs_init : np.array,
        cos_coeffs_init : np.array
        ):
        # Super
        super(FourierLatentLayer, self).__init__()

        self.freqs = self.add_weight(
            name="freqs",
            shape=freqs.shape,
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
            initializer=tf.constant_initializer(cos_coeffs_init),
            dtype='float32'
            )

        self.sin_coeff = self.add_weight(
            name="sin_coeff",
            shape=(len(freqs)),
            initializer=tf.constant_initializer(sin_coeffs_init),
            dtype='float32'
            )

    # Build function is only needed if the input is not known until runtime
    # The __call__() method of your layer will automatically run build the first 
    # time it is called. You now have a layer that's lazy and thus easier to use:
    # input_shape is the input in the call method
    # def build(self, input_shape):
    
    # @tf.function
    def fourier_range(self):
        st = tf.math.reduce_sum(abs(self.sin_coeff))
        ct = tf.math.reduce_sum(abs(self.cos_coeff))
        return tf.math.maximum(1.0, st+ct+1.e-8)

    def call(self, inputs):
        
        # tf.math.reduce_sum is same as np.sum
        # tf.matmul is same np.dot
        tsin = tf.map_fn(lambda t: tf.math.sin(t * self.freqs), inputs["t"])

        # Dot product = tf.tensordot(a, b, 1)
        ts = tf.map_fn(lambda tsinL: tf.tensordot(self.sin_coeff, tsinL, 1), tsin)

        # Same for cos
        tcos = tf.map_fn(lambda t: tf.math.cos(t * self.freqs), inputs["t"])
        tc = tf.map_fn(lambda tcosL: tf.tensordot(self.cos_coeff, tcosL, 1), tcos)

        return (self.offset_fixed + ts + tc) / self.fourier_range()

class ConvertParamsLayer(tf.keras.layers.Layer):

    def __init__(self):
        # Super
        super(ConvertParamsLayer, self).__init__()

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

class ConvertParamsLayerFrom0(tf.keras.layers.Layer):

    def __init__(self):
        # Super
        super(ConvertParamsLayerFrom0, self).__init__()

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

class ConvertParams0ToParamsLayer(tf.keras.layers.Layer):

    def __init__(self,
        nv : int,
        nh : int,
        freqs : np.array,
        muh_sin_coeffs_init : np.array,
        muh_cos_coeffs_init : np.array,
        varh_sin_coeffs_init : np.array,
        varh_cos_coeffs_init : np.array
        ):
        # Super
        super(ConvertParams0ToParamsLayer, self).__init__()

        self.nh = nh
        self.layer_muh = {}
        self.layer_varh_diag = {}
        for ih in range(0,nh):

            # Only use strings as keys
            # https://stackoverflow.com/a/57974800/1427316
            self.layer_muh[str(ih)] = FourierLatentLayer(
                freqs=freqs,
                offset_fixed=0.0,
                sin_coeffs_init=muh_sin_coeffs_init,
                cos_coeffs_init=muh_cos_coeffs_init
                )

            self.layer_varh_diag[str(ih)] = FourierLatentLayer(
                freqs=freqs,
                offset_fixed=1.01,
                sin_coeffs_init=varh_sin_coeffs_init,
                cos_coeffs_init=varh_cos_coeffs_init
                )

        self.convert_from_0 = ConvertParamsLayerFrom0()

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

class ConvertParamsToMomentsLayer(tf.keras.layers.Layer):

    def __init__(self,
        nv : int,
        nh : int
        ):
        # Super
        super(ConvertParamsToMomentsLayer, self).__init__()
        self.nv = nv 
    
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

class ConvertMomentsToNMomentsLayer(tf.keras.layers.Layer):

    def __init__(self):
        # Super
        super(ConvertMomentsToNMomentsLayer, self).__init__()
    
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

class ConvertParams0ToNMomentsLayer(tf.keras.layers.Layer):

    def __init__(self, 
        nv: int, 
        nh: int,
        freqs : np.array,
        muh_sin_coeffs_init : np.array,
        muh_cos_coeffs_init : np.array,
        varh_sin_coeffs_init : np.array,
        varh_cos_coeffs_init : np.array
        ):
        # Super
        super(ConvertParams0ToNMomentsLayer, self).__init__()

        self.params0ToParamsLayer = ConvertParams0ToParamsLayer(
            nv=nv, 
            nh=nh, 
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            varh_sin_coeffs_init=varh_sin_coeffs_init,
            varh_cos_coeffs_init=varh_cos_coeffs_init
            )
        self.paramsToMomentsLayer = ConvertParamsToMomentsLayer(nv,nh)
        self.momentsToNMomentsLayer = ConvertMomentsToNMomentsLayer()

    def call(self, inputs):
        params = self.params0ToParamsLayer(inputs)
        moments = self.paramsToMomentsLayer(params)
        nmoments = self.momentsToNMomentsLayer(moments)

        dall = {}
        for d in [params,moments,nmoments]:
            dall.update(d)
        
        return dall

# @tf.function
def unit_mat_sym(n: int, i: int, j: int):
    idx = i * n + j
    one_hot = tf.one_hot(indices=idx,depth=n*n, dtype='float32')
    
    if i != j:
        idx = j * n + i
        one_hot += tf.one_hot(indices=idx,depth=n*n, dtype='float32')

    return tf.reshape(one_hot,shape=(n,n))

class DeathRxnLayer(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, i_sp: int):
        # Super
        super(DeathRxnLayer, self).__init__()
    
        self.nv = nv
        self.nh = nh
        self.n = nv + nh
        self.i_sp = i_sp

    def call(self, inputs):

        mu = inputs["mu"]
        nvar = inputs["nvar"]

        unit = tf.one_hot(
            indices=self.i_sp,
            depth=self.n
            )

        muTE = tf.map_fn(lambda muL: - muL[self.i_sp] * unit, mu)
        
        nvarTE = tf.zeros(shape=nvar.shape, dtype='float32')
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

class BirthRxnLayer(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, i_sp: int):
        # Super
        super(BirthRxnLayer, self).__init__()
    
        self.nv = nv
        self.nh = nh
        self.n = nv + nh
        self.i_sp = i_sp

    def call(self, inputs):
        
        mu = inputs["mu"]
        nvar = inputs["nvar"]

        unit = tf.one_hot(
            indices=self.i_sp,
            depth=self.n
            )

        muTE = tf.map_fn(lambda muL: muL[self.i_sp] * unit, mu)

        nvarTE = tf.zeros(shape=nvar.shape, dtype='float32')
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
def nmoment3(mu, nvar, i, j, k):
    return -2.0*mu[i]*mu[j]*mu[k] + mu[i]*nvar[j,k] + mu[j]*nvar[i,k] + mu[k]*nvar[i,j]

class EatRxnLayer(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int, i_hunter: int, i_prey: int):
        # Super
        super(EatRxnLayer, self).__init__()
    
        self.nv = nv
        self.nh = nh
        self.n = nv + nh
        self.i_hunter = i_hunter
        self.i_prey = i_prey

    # @tf.function
    def nvar_prey_prey(self, mu, nvar):
        n3 = nmoment3(mu,nvar,self.i_prey,self.i_prey,self.i_hunter)
        return unit_mat_sym(self.n, self.i_prey, self.i_prey) * ( -2.0*n3 + nvar[self.i_hunter, self.i_prey] )

    # @tf.function
    def nvar_hunter_hunter(self, mu, nvar):
        n3 = nmoment3(mu,nvar,self.i_hunter,self.i_hunter,self.i_prey)
        return unit_mat_sym(self.n, self.i_hunter, self.i_hunter) * ( 2.0*n3 + nvar[self.i_hunter, self.i_prey] )

    # @tf.function
    def nvar_hunter_prey(self, mu, nvar):
        n3hhp = nmoment3(mu,nvar,self.i_hunter,self.i_hunter,self.i_prey)
        n3hpp = nmoment3(mu,nvar,self.i_hunter,self.i_prey,self.i_prey)
        return unit_mat_sym(self.n, self.i_hunter, self.i_prey) * ( - n3hhp + n3hpp - nvar[self.i_hunter, self.i_prey] )

    # @tf.function
    def nvar_loop(self, mu, nvar, j : int):
        um_prey = unit_mat_sym(self.n, self.i_prey, j)
        um_hunter = unit_mat_sym(self.n, self.i_hunter, j)
        n3 = nmoment3(mu,nvar,j,self.i_prey,self.i_hunter)
        return (um_hunter - um_prey) * n3

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
        
        muTE = - nvar[self.i_hunter, self.i_prey] * unit_prey + nvar[self.i_hunter,self.i_prey] * unit_hunter
        
        nvarTE = tf.zeros(shape=(self.n,self.n), dtype='float32')

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

class ConvertNMomentsTEtoMomentsTE(tf.keras.layers.Layer):

    def __init__(self):
        # Super
        super(ConvertNMomentsTEtoMomentsTE, self).__init__()

    def call(self, inputs):
        
        mu = inputs["mu"]
        muTE = inputs["muTE"]
        nvarTE = inputs["nvarTE"]

        # kronecker product of two vectors = tf.tensordot(a,b,axes=0)
        neg_kpv = - tf.tensordot(muTE,mu,axes=0) - tf.tensordot(mu,muTE,axes=0)

        varTE = nvarTE + neg_kpv

        return {
            "muTE": muTE,
            "varTE": varTE
        }

class ConvertMomentsTEtoParamMomentsTE(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int):
        # Super
        super(ConvertMomentsTEtoParamMomentsTE, self).__init__()
    
        self.nv = nv
        self.nh = nh

    def call(self, inputs):
        
        muTE = inputs["muTE"]
        varTE = inputs["varTE"]

        muvTE = muTE[:self.nv]
        muhTE = muTE[self.nv:]
        varvhTE = varTE[self.nv:,:self.nv]
        varvbarTE = tf.linalg.trace(varTE[:self.nv,:self.nv])
        varh_diagTE = tf.linalg.diag_part(varTE[self.nv:,self.nv:])

        return {
            "muvTE": muvTE,
            "muhTE": muhTE,
            "varvhTE": varvhTE,
            "varvbarTE": varvbarTE,
            "varh_diagTE": varh_diagTE
        }

class ConvertParamMomentsTEtoParamsTE(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int):
        # Super
        super(ConvertParamMomentsTEtoParamsTE, self).__init__()
    
        self.nv = nv
        self.nh = nh

    def call(self, inputs):

        muvTE = inputs["muvTE"]
        varvhTE = inputs["varvhTE"]
        varh_diagTE = inputs["varh_diagTE"]
        varh_diag = inputs["varh_diag"]
        muh = inputs["muh"]
        varvh = inputs["varvh"]
        muhTE = inputs["muhTE"]
        varvbarTE = inputs["varvbarTE"]

        varh_inv = tf.linalg.tensor_diag(1.0 / varh_diag)
        varhTE = tf.linalg.tensor_diag(varh_diagTE)

        bTE = muvTE 
        bTE -= tf.linalg.matvec(tf.transpose(varvhTE), 
            tf.linalg.matvec(varh_inv, muh))
        bTE += tf.linalg.matvec(tf.transpose(varvh), 
            tf.linalg.matvec(varh_inv, 
            tf.linalg.matvec(varhTE, 
            tf.linalg.matvec(varh_inv, muh))))
        bTE -= tf.linalg.matvec(tf.transpose(varvh),
            tf.linalg.matvec(varh_inv, muhTE))
        
        wtTE = - tf.matmul(varh_inv,
            tf.matmul(varhTE,
            tf.matmul(varh_inv,varvh)))
        wtTE += tf.matmul(varh_inv, varvhTE)

        sig2TEmat = tf.matmul(tf.transpose(varvhTE),
            tf.matmul(varh_inv, varvh))
        sig2TEmat -= tf.matmul(tf.transpose(varvh),
            tf.matmul(varh_inv,
            tf.matmul(varhTE,
            tf.matmul(varh_inv,varvh))))
        sig2TEmat += tf.matmul(tf.transpose(varvh),
            tf.matmul(varh_inv,varvhTE))
        sig2TEtr = tf.linalg.trace(sig2TEmat)
        sig2TE = (varvbarTE - sig2TEtr) / self.nv

        return {
            "bTE": bTE,
            "wtTE": wtTE,
            "sig2TE": sig2TE,
            "muhTE": muhTE,
            "varh_diagTE": varh_diagTE
        }

class ConvertParamsTEtoParams0TE(tf.keras.layers.Layer):

    def __init__(self):
        # Super
        super(ConvertParamsTEtoParams0TE, self).__init__()

    def call(self, inputs):

        bTE1 = inputs["bTE1"]
        wtTE1 = inputs["wtTE1"]
        muh1 = inputs["muh1"]
        wt1 = inputs["wt1"]
        muhTE1 = inputs["muhTE1"]
        varh_diag1 = inputs["varh_diag1"]
        varh_diagTE1 = inputs["varh_diagTE1"]
        sig2TE = inputs["sig2TE"]

        varh1 = tf.linalg.tensor_diag(varh_diag1)
        varh_inv1 = tf.linalg.tensor_diag(1.0 / varh_diag1)
        varhTE1 = tf.linalg.tensor_diag(varh_diagTE1)

        wTE1 = tf.transpose(wtTE1)
        w1 = tf.transpose(wt1)

        bTE2 = bTE1 + tf.linalg.matvec(wTE1,muh1) + tf.linalg.matvec(w1,muhTE1)
        wtTE2 = 0.5 * tf.matmul(tf.math.sqrt(varh_inv1),
            tf.matmul(varhTE1, wt1)) + tf.matmul(tf.math.sqrt(varh1),wtTE1)
        
        return {
            "sig2TE": sig2TE,
            "bTE2" : bTE2,
            "wtTE2" : wtTE2
        }

class ConvertNMomentsTEtoParams0TE(tf.keras.layers.Layer):

    def __init__(self, nv: int, nh: int):
        # Super
        super(ConvertNMomentsTEtoParams0TE, self).__init__()
        
        self.nv = nv
        self.nh = nh

        self.nMomentsTEtoMomentsTE = ConvertNMomentsTEtoMomentsTE()
        self.momentsTEtoParamMomentsTE = ConvertMomentsTEtoParamMomentsTE(nv,nh)
        self.paramMomentsTEtoParamsTE = ConvertParamMomentsTEtoParamsTE(nv,nh)
        self.paramsTEtoParams0TE = ConvertParamsTEtoParams0TE()

    def call(self, inputs):
        outputs1 = self.nMomentsTEtoMomentsTE(inputs)

        outputs2 = self.momentsTEtoParamMomentsTE(outputs1)

        outputs2["varh_diag"] = inputs["varh_diag"]
        outputs2["muh"] = inputs["muh"]
        outputs2["varvh"] = inputs["var"][self.nv:,:self.nv]
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

class RxnSpec(Enum):
    BIRTH = 0
    DEATH = 1
    EAT = 2

# Model vs Layer
# https://www.tensorflow.org/tutorials/customization/custom_layers#models_composing_layers
# Typically you inherit from keras.Model when you need the model methods like: Model.fit,Model.evaluate, and Model.save (see Custom Keras layers and models for details).
# One other feature provided by keras.Model (instead of keras.layers.Layer) is that in addition to tracking variables, a keras.Model also tracks its internal layers, making them easier to inspect.
class RxnInputsLayer(tf.keras.layers.Layer):

    def __init__(self, 
        nv: int, 
        nh: int, 
        freqs : np.array,
        muh_sin_coeffs_init : np.array,
        muh_cos_coeffs_init : np.array,
        varh_sin_coeffs_init : np.array,
        varh_cos_coeffs_init : np.array,
        rxn_specs : List[Union[Tuple[RxnSpec,int],Tuple[RxnSpec,int,int]]]
        ):
        # Super
        super(RxnInputsLayer, self).__init__(name="rxn_inputs")
        
        self.params0toNMoments = ConvertParams0ToNMomentsLayer(
            nv=nv,
            nh=nh,
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            varh_sin_coeffs_init=varh_sin_coeffs_init,
            varh_cos_coeffs_init=varh_cos_coeffs_init
            )

        self.rxns = []
        for spec in rxn_specs:
            if spec[0] == RxnSpec.EAT:
                rtype, i_hunter, i_prey = spec
            else:
                rtype, i_sp = spec

            if rtype == RxnSpec.BIRTH:
                self.rxns.append(BirthRxnLayer(nv, nh, i_sp))
            elif rtype == RxnSpec.DEATH:
                self.rxns.append(DeathRxnLayer(nv, nh, i_sp))
            elif rtype == RxnSpec.EAT:
                self.rxns.append(EatRxnLayer(nv,nh,i_hunter,i_prey))
            else:
                raise ValueError("Rxn type: %s not recognized" % rtype)

        self.nMomentsTEtoParams0TE = ConvertNMomentsTEtoParams0TE(nv,nh)

    def call(self, inputs):
        nMoments = self.params0toNMoments(inputs)

        # Compute reactions
        params0TEforRxns = []
        for i in range(0,len(self.rxns)):
            nMomentsTE = self.rxns[i](nMoments)

            # Convert nMomentsTE to params0TE
            nMomentsTE.update(nMoments)
            params0TE = self.nMomentsTEtoParams0TE(nMomentsTE)

            # Flatten
            params0TEarr = [tf.reshape(val, [-1]) for val in params0TE.values()]
            params0TEflat = tf.concat(params0TEarr, 0)

            # Store
            params0TEforRxns.append(params0TEflat)

        # Flatten all reactions
        params0TE = tf.concat(params0TEforRxns, 0)

        return params0TE