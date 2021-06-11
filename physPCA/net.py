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