from physPCA import FourierLatent

import numpy as np
import tensorflow as tf

class TestFourier:

    def test_fourier(self):

        freqs = np.random.rand(3)
        sin_coeffs_init = np.random.rand(3)
        cos_coeffs_init = np.random.rand(3)

        # Create layer
        fl = FourierLatent(
            freqs=freqs,
            offset_fixed=0.0,
            sin_coeffs_init=sin_coeffs_init,
            cos_coeffs_init=cos_coeffs_init
            )

        # Input
        x = tf.constant(3, dtype='float32')
        
        # Output
        y = fl(x)

        st = tf.math.reduce_sum(abs(self.sin_coeff))
        ct = tf.math.reduce_sum(abs(self.cos_coeff))
        return tf.math.maximum(1.0, st+ct+1.e-8)
        
        print(y)