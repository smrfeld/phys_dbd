from physDBD import RxnSpec, FourierLatentLayer, \
    ConvertParamsLayer, ConvertParamsLayerFrom0, ConvertParams0ToParamsLayer, \
        ConvertParamsToMomentsLayer, ConvertMomentsToNMomentsLayer, DeathRxnLayer, BirthRxnLayer, EatRxnLayer, \
            ConvertNMomentsTEtoMomentsTE, ConvertMomentsTEtoParamMomentsTE, ConvertParamMomentsTEtoParamsTE, \
                ConvertParamsTEtoParams0TE, ConvertNMomentsTEtoParams0TE, ConvertParams0ToNMomentsLayer, RxnInputsLayer

import numpy as np
import tensorflow as tf

class TestNet:

    def get_random_var(self, batch_size: int, nv: int, nh: int):
        nvar = np.random.rand(batch_size,nv+nh,nv+nh)
        nvar += np.transpose(nvar,axes=[0,2,1])
        return nvar

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
        batch_size = 1
        x_in = {
            "t": tf.reshape(tf.constant(3, dtype='float32'), shape=(batch_size, 1))
        }
        
        # Output
        x_out = fl(x_in)

        print(x_out)

    def test_convert_params_layer(self):

        lyr = ConvertParamsLayer()

        nv = 3
        nh = 2
        batch_size = 2
        x_in = {
            "b1": tf.constant(np.random.rand(batch_size,nv), dtype="float32"),
            "wt1": tf.constant(np.random.rand(batch_size,nh,nv), dtype="float32"),
            "muh1": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "muh2": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "varh_diag1": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "varh_diag2": tf.constant(np.random.rand(batch_size,nh), dtype="float32")
            }

        x_out = lyr(x_in)
        
        print(x_out)

    def test_convert_from_0(self):

        lyr = ConvertParamsLayerFrom0()

        nv = 3
        nh = 2
        batch_size = 2
        x_in = {
            "b1": tf.constant(np.random.rand(batch_size,nv), dtype="float32"),
            "wt1": tf.constant(np.random.rand(batch_size,nh,nv), dtype="float32"),
            "muh2": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "varh_diag2": tf.constant(np.random.rand(batch_size,nh), dtype="float32")
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
        batch_size = 2
        x_in = {
            "t": tf.constant(np.arange(3,3+batch_size), dtype='float32'),
            "b": tf.constant(np.random.rand(batch_size,nv), dtype="float32"),
            "wt": tf.constant(np.random.rand(batch_size,nh,nv), dtype="float32"),
            "sig2": tf.constant(np.random.rand(batch_size), dtype='float32')
            }   
             
        # Output
        x_out = lyr(x_in)

        print(x_out)
    
    def test_params_to_moments(self):

        nv = 3
        nh = 2

        lyr = ConvertParamsToMomentsLayer(
            nv=nv,
            nh=nh
        )

        # Input
        batch_size = 2
        x_in = {
            "b": tf.constant(np.random.rand(batch_size,nv), dtype="float32"),
            "wt": tf.constant(np.random.rand(batch_size,nh,nv), dtype="float32"),
            "sig2": tf.constant(np.random.rand(batch_size), dtype='float32'),
            "varh_diag": tf.constant(np.random.rand(batch_size,nh), dtype='float32'),
            "muh": tf.constant(np.random.rand(batch_size,nh), dtype='float32')
            }   
             
        # Output
        x_out = lyr(x_in)

        print(x_out)

    def test_moments_to_nmoments(self):

        lyr = ConvertMomentsToNMomentsLayer()

        nv = 3
        nh = 2

        # Input
        batch_size = 2
        x_in = {
            "mu": tf.constant(np.random.rand(batch_size,nv+nh), dtype="float32"),
            "var": tf.constant(self.get_random_var(batch_size,nv,nh), dtype="float32")
            }   
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

    def test_params0_to_nmoments(self):

        nv = 3
        nh = 2

        freqs = np.random.rand(3)
        muh_sin_coeffs_init = np.random.rand(3)
        muh_cos_coeffs_init = np.random.rand(3)
        varh_sin_coeffs_init = np.random.rand(3)
        varh_cos_coeffs_init = np.random.rand(3)

        lyr = ConvertParams0ToNMomentsLayer(
            nv=nv,
            nh=nh,
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            varh_sin_coeffs_init=varh_sin_coeffs_init,
            varh_cos_coeffs_init=varh_cos_coeffs_init
        )

        # Input
        batch_size = 2
        x_in = {
            "t": tf.constant(np.arange(4,4+batch_size), dtype='float32'),
            "b": tf.constant(np.random.rand(batch_size,nv), dtype="float32"),
            "wt": tf.constant(np.random.rand(batch_size,nh,nv), dtype="float32"),
            "sig2": tf.constant(np.random.rand(batch_size), dtype='float32')
            }
             
        # Output
        x_out = lyr(x_in)
        
        print(x_out)
    
    def test_death_rxn(self):

        nv = 3
        nh = 2
        lyr = DeathRxnLayer(nv=nv,nh=nh,i_sp=1)

        # Input
        batch_size = 2
        x_in = {
            "mu": tf.constant(np.random.rand(batch_size,nv+nh), dtype="float32"),
            "nvar": tf.constant(self.get_random_var(batch_size,nv,nh), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

    def test_birth_rxn(self):

        nv = 3
        nh = 2
        lyr = BirthRxnLayer(nv=nv,nh=nh,i_sp=1)

        # Input
        batch_size = 2
        x_in = {
            "mu": tf.constant(np.random.rand(batch_size,nv+nh), dtype="float32"),
            "nvar": tf.constant(self.get_random_var(batch_size,nv,nh), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

    def test_eat_rxn(self):

        nv = 3
        nh = 2
        lyr = EatRxnLayer(nv=nv,nh=nh,i_prey=1,i_hunter=2)

        # Input
        batch_size = 2
        x_in = {
            "mu": tf.constant(np.random.rand(batch_size,nv+nh), dtype="float32"),
            "nvar": tf.constant(self.get_random_var(batch_size,nv,nh), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

    def test_convert_nmomentsTE_to_momentsTE(self):
        nv = 3
        nh = 2
        lyr = ConvertNMomentsTEtoMomentsTE()

        # Input
        x_in = {
            "mu": tf.constant(np.random.rand(nv+nh), dtype="float32"),
            "muTE": tf.constant(np.random.rand(nv+nh), dtype="float32"),
            "nvarTE": tf.constant(self.get_random_var(batch_size,nv,nh), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

    def test_convert_momentsTE_to_paramMomentsTE(self):
        nv = 3
        nh = 2
        lyr = ConvertMomentsTEtoParamMomentsTE(nv=nv,nh=nh)

        # Input
        x_in = {
            "muTE": tf.constant(np.random.rand(nv+nh), dtype="float32"),
            "varTE": tf.constant(self.get_random_var(batch_size,nv,nh), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

    def test_convert_paramMomentsTE_to_paramsTE(self):
        nv = 3
        nh = 2
        lyr = ConvertParamMomentsTEtoParamsTE(nv=nv,nh=nh)

        # Input
        x_in = {
            "muvTE": tf.constant(np.random.rand(nv), dtype="float32"),
            "varvhTE": tf.constant(np.random.rand(nh,nv), dtype="float32"),
            "varh_diagTE": tf.constant(np.random.rand(nh), dtype="float32"),
            "varh_diag": tf.constant(np.random.rand(nh), dtype="float32"),
            "muh": tf.constant(np.random.rand(nh), dtype="float32"),
            "varvh": tf.constant(np.random.rand(nh,nv), dtype="float32"),
            "muhTE": tf.constant(np.random.rand(nh), dtype="float32"),
            "varvbarTE": tf.constant(np.random.rand(), dtype="float32")
            }
        
        # Output
        x_out = lyr(x_in)

        print(x_out)

    def test_convert_paramsTE_to_params0TE(self):
        lyr = ConvertParamsTEtoParams0TE()

        nv = 3
        nh = 2
        x_in = {
            "bTE1": tf.constant(np.random.rand(nv), dtype="float32"),
            "wtTE1": tf.constant(np.random.rand(nh,nv), dtype="float32"),
            "muh1": tf.constant(np.random.rand(nh), dtype="float32"),
            "wt1": tf.constant(np.random.rand(nh,nv), dtype="float32"),
            "muhTE1": tf.constant(np.random.rand(nh), dtype="float32"),
            "varh_diag1": tf.constant(np.random.rand(nh), dtype="float32"),
            "varh_diagTE1": tf.constant(np.random.rand(nh), dtype="float32"),
            "sig2TE": tf.constant(np.random.rand(), dtype="float32")
        }
                
        # Output
        x_out = lyr(x_in)

        print(x_out)

    def test_convert_nmomentsTE_to_params0TE(self):

        nv = 3
        nh = 2
        lyr = ConvertNMomentsTEtoParams0TE(nv,nh)

        # Input
        x_in = {
            "mu": tf.constant(np.random.rand(nv+nh), dtype="float32"),
            "muTE": tf.constant(np.random.rand(nv+nh), dtype="float32"),
            "nvarTE": tf.constant(self.get_random_var(batch_size,nv,nh), dtype="float32"),
            "varh_diag": tf.constant(np.random.rand(nh), dtype="float32"),
            "muh": tf.constant(np.random.rand(nh), dtype="float32"),
            "var": tf.constant(self.get_random_var(batch_size,nv,nh), dtype="float32"),
            "wt": tf.constant(np.random.rand(nh,nv), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

    def test_rxn_inputs(self):

        nv = 3
        nh = 2

        freqs = np.random.rand(3)
        muh_sin_coeffs_init = np.random.rand(3)
        muh_cos_coeffs_init = np.random.rand(3)
        varh_sin_coeffs_init = np.random.rand(3)
        varh_cos_coeffs_init = np.random.rand(3)

        rxn_specs = [
            (RxnSpec.BIRTH,0),
            (RxnSpec.DEATH,1),
            (RxnSpec.EAT,2,1)
        ]

        lyr = RxnInputsLayer(
            nv=nv,
            nh=nh,
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            varh_sin_coeffs_init=varh_sin_coeffs_init,
            varh_cos_coeffs_init=varh_cos_coeffs_init,
            rxn_specs=rxn_specs
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