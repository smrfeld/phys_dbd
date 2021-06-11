from physPCA import FourierLatentLayer, ConvertParamsLayer

import numpy as np
import tensorflow as tf

class TestNet:

    def stest_fourier(self):

        freqs = np.random.rand(3)
        sin_coeffs_init = np.random.rand(3)
        cos_coeffs_init = np.random.rand(3)

        # Create layer
        fl = FourierLatentLayer(
            freqs=freqs,
            offset_fixed=0.0,
            sin_coeffs_init=sin_coeffs_init,
            cos_coeffs_init=cos_coeffs_init
            )

        # Input
        x_in = {
            "t": tf.constant(3, dtype='float32')
        }
        
        # Output
        x_out = fl(x_in)

        print(x_out)

    def test_convert(self):

        lyr = ConvertParamsLayer()

        nv = 3
        nh = 2
        x_in = {
            "b1": tf.constant(np.random.rand(nv), dtype="float32"),
            "wt1": tf.constant(np.random.rand(nh,nv), dtype="float32"),
            "muh1": tf.constant(np.random.rand(nh), dtype="float32"),
            "muh2": tf.constant(np.random.rand(nh), dtype="float32"),
            "varh_diag1": tf.constant(np.random.rand(nh), dtype="float32"),
            "varh_diag2": tf.constant(np.random.rand(nh), dtype="float32")
            }

        x_out = lyr(x_in)
        
        print(x_out)