import tensorflow as tf

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
        cov = inputs["cov"]

        # kronecker product of two vectors = tf.tensordot(a,b,axes=0)
        kpv = tf.map_fn(lambda muL: tf.tensordot(muL,muL,axes=0),mu)

        ncov = cov + kpv

        return {
            "mu": mu,
            "ncov": ncov
        }

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
        ncov = inputs["ncov"]

        unit = tf.one_hot(
            indices=self.i_sp,
            depth=self.n
            )

        muTE = tf.map_fn(lambda muL: - muL[self.i_sp] * unit, mu)
        
        # tf.zeros_like vs tf.zeros
        # https://stackoverflow.com/a/49599952/1427316
        # avoid 'Cannot convert a partially known TensorShape to a Tensor'
        ncovTE = tf.zeros_like(ncov)
        for j in range(0,self.n):
            if j == self.i_sp:
                vals = -2.0 * ncov[:,self.i_sp,self.i_sp] + mu[:,self.i_sp]
            else:
                vals = -1.0 * ncov[:,self.i_sp,j]

            unit_mat = unit_mat_sym(self.n,self.i_sp,j)
            ncovTE += tf.map_fn(lambda val: unit_mat * val, vals)
        
        return {
            "muTE": muTE,
            "ncovTE": ncovTE
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
        ncov = inputs["ncov"]

        unit = tf.one_hot(
            indices=self.i_sp,
            depth=self.n
            )

        muTE = tf.map_fn(lambda muL: muL[self.i_sp] * unit, mu)

        ncovTE = tf.zeros_like(ncov)
        for j in range(0,self.n):
            if j == self.i_sp:
                vals = 2.0 * ncov[:,self.i_sp,self.i_sp] + mu[:,self.i_sp]
            else:
                vals = ncov[:,self.i_sp,j]

            unit_mat = unit_mat_sym(self.n,self.i_sp,j)
            ncovTE += tf.map_fn(lambda val: unit_mat * val, vals)
        
        return {
            "muTE": muTE,
            "ncovTE": ncovTE
        }

# @tf.function
@tf.keras.utils.register_keras_serializable(package="physDBD")
def nmoment3_batch(mu, ncov, i, j, k):
    return -2.0*mu[:,i]*mu[:,j]*mu[:,k] + mu[:,i]*ncov[:,j,k] + mu[:,j]*ncov[:,i,k] + mu[:,k]*ncov[:,i,j]

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
    def ncov_prey_prey(self, mu, ncov):
        n3 = nmoment3_batch(mu,ncov,self.i_prey,self.i_prey,self.i_hunter)
        vals = -2.0*n3 + ncov[:, self.i_hunter, self.i_prey]
        unit_mat = unit_mat_sym(self.n, self.i_prey, self.i_prey)
        return tf.map_fn(lambda val: unit_mat * val, vals)

    # @tf.function
    def ncov_hunter_hunter(self, mu, ncov):
        n3 = nmoment3_batch(mu,ncov,self.i_hunter,self.i_hunter,self.i_prey)
        vals = 2.0*n3 + ncov[:,self.i_hunter, self.i_prey]
        unit_mat = unit_mat_sym(self.n, self.i_hunter, self.i_hunter)
        return tf.map_fn(lambda val: unit_mat * val, vals)

    # @tf.function
    def ncov_hunter_prey(self, mu, ncov):
        n3hhp = nmoment3_batch(mu,ncov,self.i_hunter,self.i_hunter,self.i_prey)
        n3hpp = nmoment3_batch(mu,ncov,self.i_hunter,self.i_prey,self.i_prey)
        unit_mat = unit_mat_sym(self.n, self.i_hunter, self.i_prey)
        vals = - n3hhp + n3hpp - ncov[:,self.i_hunter, self.i_prey] 
        return tf.map_fn(lambda val: unit_mat * val, vals)

    # @tf.function
    def ncov_loop(self, mu, ncov, j : int):
        um_prey = unit_mat_sym(self.n, self.i_prey, j)
        um_hunter = unit_mat_sym(self.n, self.i_hunter, j)
        unit_mat = um_hunter - um_prey
        n3 = nmoment3_batch(mu,ncov,j,self.i_prey,self.i_hunter)
        vals = n3
        return tf.map_fn(lambda val: unit_mat * val, vals)

    def call(self, inputs):
        
        mu = inputs["mu"]
        ncov = inputs["ncov"]

        unit_hunter = tf.one_hot(
            indices=self.i_hunter,
            depth=self.n
            )
        unit_prey = tf.one_hot(
            indices=self.i_prey,
            depth=self.n
            )
        
        muTE = tf.map_fn(
            lambda ncovL: - ncovL[self.i_hunter, self.i_prey] * unit_prey \
                + ncovL[self.i_hunter,self.i_prey] * unit_hunter,
                ncov)
        
        ncovTE = tf.zeros_like(ncov)

        # Prey-prey
        ncovTE += self.ncov_prey_prey(mu,ncov)
        
        # Hunter-hunter
        ncovTE += self.ncov_hunter_hunter(mu,ncov)

        # Hunter-prey
        ncovTE += self.ncov_hunter_prey(mu,ncov)

        # Loop
        for j in range(0,self.n):
            if j != self.i_prey and j != self.i_hunter:
                ncovTE += self.ncov_loop(mu,ncov,j)

        return {
            "muTE": muTE,
            "ncovTE": ncovTE
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
        ncovTE = inputs["ncovTE"]

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

        covTE = ncovTE + neg_kpv

        return {
            "muTE": muTE,
            "covTE": covTE
        }