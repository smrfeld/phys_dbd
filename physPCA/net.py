import tensorflow as tf

import numpy as np

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
    
    @tf.function
    def fourier_range(self):
        st = tf.math.reduce_sum(abs(self.sin_coeff))
        ct = tf.math.reduce_sum(abs(self.cos_coeff))
        return tf.math.maximum(1.0, st+ct+1.e-8)

    def call(self, inputs):
        # tf.math.reduce_sum is same as np.sum
        # tf.matmul is same np.dot
        tsin = tf.math.sin(inputs["t"] * self.freqs)

        # Dot product = tf.tensordot(a, b, 1)
        ts = tf.tensordot(self.sin_coeff, tsin, 1)

        # Same for cos
        tcos = tf.math.cos(inputs["t"] * self.freqs)
        tc = tf.tensordot(self.cos_coeff, tcos, 1)

        return (self.offset_fixed + ts + tc) / self.fourier_range()

class ConvertParamsLayer(tf.keras.layers.Layer):

    def __init__(self):
        # Super
        super(ConvertParamsLayer, self).__init__()

    def call(self, inputs):

        b1 = inputs["b1"]
        wt1 = inputs["wt1"]
        w1 = tf.transpose(wt1)
        muh1 = inputs["muh1"]
        muh2 = inputs["muh2"]
        varh_diag1 = inputs["varh_diag1"]
        varh_diag2 = inputs["varh_diag2"]

        varh1_sqrt = tf.math.sqrt(tf.linalg.tensor_diag(varh_diag1))
        varh2_inv_sqrt = tf.linalg.tensor_diag(tf.math.sqrt(tf.math.pow(varh_diag2,-1)))

        # Matrix * vector = tf.linalg.matvec
        # Diagonal matrix = tf.linalg.tensor_diag
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
        w1 = tf.transpose(wt1)
        muh2 = inputs["muh2"]
        varh_diag2 = inputs["varh_diag2"]

        varh2_inv_sqrt = tf.linalg.tensor_diag(tf.math.sqrt(tf.math.pow(varh_diag2,-1)))

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
            self.layer_muh[ih] = FourierLatentLayer(
                freqs=freqs,
                offset_fixed=0.0,
                sin_coeffs_init=muh_sin_coeffs_init,
                cos_coeffs_init=muh_cos_coeffs_init
                )

            self.layer_varh_diag[ih] = FourierLatentLayer(
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
            muhs.append(self.layer_muh[ih](inputs))
            varh_diags.append(self.layer_varh_diag[ih](inputs))

        muh = tf.concat(muhs,0)
        varh_diag = tf.concat(varh_diags,0)

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

class MomentsFromParamsLayer(tf.keras.layers.Layer):

    def __init__(self,
        nv : int,
        nh : int
        ):
        # Super
        super(MomentsFromParamsLayer, self).__init__()
        self.nv = nv 
    
    def call(self, inputs):

        wt = inputs["wt"]
        varh_diag = inputs["varh_diag"]
        sig2 = inputs["sig2"]
        b = inputs["b"]
        muh = inputs["muh"]

        w = tf.transpose(wt)
        varh = tf.linalg.tensor_diag(varh_diag)

        varvh = tf.matmul(varh, wt)
        varv = tf.matmul(w,tf.matmul(varh,wt)) + sig2 * tf.eye(self.nv)

        muv = b + tf.linalg.matvec(w, muh)
        mu = tf.concat([muv,muh],0)

        varvht = tf.transpose(varvh)
        var = tf.concat([tf.concat([varv,varvht],1),tf.concat([varvh,varh],1)],0)

        return {
            "mu": mu,
            "var": var
        }

class MomentsToNMomentsLayer(tf.keras.layers.Layer):

    def __init__(self):
        # Super
        super(MomentsToNMomentsLayer, self).__init__()
    
    def call(self, inputs):
        
        mu = inputs["mu"]
        var = inputs["var"]

        # kronecker product of two vectors = tf.tensordot(a,b,axes=0)
        kpv = tf.tensordot(mu,mu,axes=0)

        nvar = var + kpv

        return {
            "mu": mu,
            "nvar": nvar
        }

@tf.function
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
        
        unit = tf.one_hot(
            indices=self.i_sp,
            depth=self.n
            )

        mu = inputs["mu"]
        nvar = inputs["var"]

        muTE = - mu[self.i_sp] * unit
        
        nvarTE = tf.zeros(shape=(self.n,self.n), dtype='float32')
        for j in range(0,self.n):
            if j == self.i_sp:
                nvarTE += unit_mat_sym(self.n,self.i_sp,self.i_sp) * (-2.0 * nvar[self.i_sp,self.i_sp] + mu[self.i_sp])
            else:
                nvarTE += unit_mat_sym(self.n,self.i_sp,j) * (-1.0 * nvar[self.i_sp,j])
        
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
        
        unit = tf.one_hot(
            indices=self.i_sp,
            depth=self.n
            )

        mu = inputs["mu"]
        nvar = inputs["var"]

        muTE = mu[self.i_sp] * unit
        
        nvarTE = tf.zeros(shape=(self.n,self.n), dtype='float32')
        for j in range(0,self.n):
            if j == self.i_sp:
                nvarTE += unit_mat_sym(self.n,self.i_sp,self.i_sp) * (2.0 * nvar[self.i_sp,self.i_sp] + mu[self.i_sp])
            else:
                nvarTE += unit_mat_sym(self.n,self.i_sp,j) * (nvar[self.i_sp,j])
        
        return {
            "muTE": muTE,
            "nvarTE": nvarTE
        }

@tf.function
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

    @tf.function
    def nvar_prey_prey(self, mu, nvar):
        n3 = nmoment3(mu,nvar,self.i_prey,self.i_prey,self.i_hunter)
        return unit_mat_sym(self.n, self.i_prey, self.i_prey) * ( -2.0*n3 + nvar[self.i_hunter, self.i_prey] )

    @tf.function
    def nvar_hunter_hunter(self, mu, nvar):
        n3 = nmoment3(mu,nvar,self.i_hunter,self.i_hunter,self.i_prey)
        return unit_mat_sym(self.n, self.i_hunter, self.i_hunter) * ( 2.0*n3 + nvar[self.i_hunter, self.i_prey] )

    @tf.function
    def nvar_hunter_prey(self, mu, nvar):
        n3hhp = nmoment3(mu,nvar,self.i_hunter,self.i_hunter,self.i_prey)
        n3hpp = nmoment3(mu,nvar,self.i_hunter,self.i_prey,self.i_prey)
        return unit_mat_sym(self.n, self.i_hunter, self.i_prey) * ( - n3hhp + n3hpp - nvar[self.i_hunter, self.i_prey] )

    @tf.function
    def nvar_loop(self, mu, nvar, j : int):
        um_prey = unit_mat_sym(self.n, self.i_prey, j)
        um_hunter = unit_mat_sym(self.n, self.i_hunter, j)
        n3 = nmoment3(mu,nvar,j,self.i_prey,self.i_hunter)
        return (um_hunter - um_prey) * n3

    def call(self, inputs):
        
        mu = inputs["mu"]
        nvar = inputs["var"]

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
