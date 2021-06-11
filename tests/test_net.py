from physPCA import FourierLatentLayer, \
    ConvertParamsLayer, ConvertParamsLayerFrom0, ConvertParams0ToParamsLayer, MomentsFromParamsLayer

import numpy as np
import tensorflow as tf

class TestNet:

    def test_fourier(self):

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

    def test_convert_from_0(self):

        lyr = ConvertParamsLayerFrom0()

        nv = 3
        nh = 2
        x_in = {
            "b1": tf.constant(np.random.rand(nv), dtype="float32"),
            "wt1": tf.constant(np.random.rand(nh,nv), dtype="float32"),
            "muh2": tf.constant(np.random.rand(nh), dtype="float32"),
            "varh_diag2": tf.constant(np.random.rand(nh), dtype="float32")
            }

        x_out = lyr(x_in)
        
        print(x_out)

    def test_convert_params0_to_params(self):

        freqs = np.random.rand(3)
        muh_sin_coeffs_init = np.random.rand(3)
        muh_cos_coeffs_init = np.random.rand(3)
        varh_sin_coeffs_init = np.random.rand(3)
        varh_cos_coeffs_init = np.random.rand(3)

        nv = 3
        nh = 2

        lyr = ConvertParams0ToParamsLayer(
            nv=nv,
            nh=nh,
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            varh_sin_coeffs_init=varh_sin_coeffs_init,
            varh_cos_coeffs_init=varh_cos_coeffs_init
        )


        # Input
        x_in = {
            "t": tf.constant(3, dtype='float32'),
            "b": tf.constant(np.random.rand(nv), dtype="float32"),
            "wt": tf.constant(np.random.rand(nh,nv), dtype="float32"),
            "sig2": tf.constant(np.random.rand(), dtype='float32')
            }   
             
        # Output
        x_out = lyr(x_in)

        print(x_out)
    
    def test_moments_from_params(self):

        nv = 3
        nh = 2

        lyr = MomentsFromParamsLayer(
            nv=nv,
            nh=nh
        )

        # Input
        x_in = {
            "b": tf.constant(np.random.rand(nv), dtype="float32"),
            "wt": tf.constant(np.random.rand(nh,nv), dtype="float32"),
            "sig2": tf.constant(np.random.rand(), dtype='float32'),
            "varh_diag": tf.constant(np.random.rand(nh), dtype='float32'),
            "muh": tf.constant(np.random.rand(nh), dtype='float32')
            }   
             
        # Output
        x_out = lyr(x_in)

        print(x_out)