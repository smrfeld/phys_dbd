from physDBD import RxnSpec, FourierLatentLayer, \
    ConvertParamsLayer, ConvertParamsLayerFrom0, ConvertParams0ToParamsLayer, \
        ConvertParamsToMomentsLayer, ConvertMomentsToNMomentsLayer, DeathRxnLayer, BirthRxnLayer, EatRxnLayer, \
            ConvertNMomentsTEtoMomentsTE, ConvertMomentsTEtoParamMomentsTE, ConvertParamMomentsTEtoParamsTE, \
                ConvertParamsTEtoParams0TE, ConvertNMomentsTEtoParams0TE, ConvertParams0ToNMomentsLayer, RxnInputsLayer

# Depreciation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import numpy as np
import tensorflow as tf

class Vals:

    nv = 3
    nh = 2
    batch_size = 2

    _b = np.array([3.0,5.0,6.0])
    _wt = np.array([[2.0,4.0,8.0],[1.0,3.0,3.0]])
    _muh = np.array([4.0,8.0])
    _varh_diag = np.array([5.0,9.0])
    _sig2 = 1.0

    _muh2 = np.array([3.0,3.0])
    _varh_diag2 = np.array([1.0,8.0])

    _wt_TE = np.array([[1.0,2.0,3.0],[3.0,3.0,3.0]])
    _b_TE = np.array([0.3,0.4,0.8])
    _sig2_TE = 0.3
    _muh_TE = np.array([0.3,0.8])
    _varh_diag_TE = np.array([0.9,0.7])

    _muh_TE2 = np.array([0.7,0.9])
    _varh_diag_TE2 = np.array([0.1,0.1])

    _freqs = np.array([1.,2.,3.])
    _muh_cos_coeffs_init = np.array([1.,2.,4.])
    _muh_sin_coeffs_init = np.array([1.,5.,4.])
    _varh_cos_coeffs_init = np.array([6.,2.,4.])
    _varh_sin_coeffs_init = np.array([1.,8.,4.])

    @classmethod
    def tpt(cls):
        return np.full(cls.batch_size,3-1) # zero indexed

    @classmethod
    def freqs(cls):
        return cls._freqs

    @classmethod
    def muh_cos_coeffs_init(cls):
        return cls._muh_cos_coeffs_init

    @classmethod
    def muh_sin_coeffs_init(cls):
        return cls._muh_sin_coeffs_init

    @classmethod
    def varh_cos_coeffs_init(cls):
        return cls._varh_cos_coeffs_init

    @classmethod
    def varh_sin_coeffs_init(cls):
        return cls._varh_sin_coeffs_init

    @classmethod
    def b(cls):
        return np.tile(cls._b, (cls.batch_size,1))
    
    @classmethod
    def wt(cls):
        return np.tile(cls._wt, (cls.batch_size,1,1))

    @classmethod
    def muh(cls):
        return np.tile(cls._muh, (cls.batch_size,1))

    @classmethod
    def varh_diag(cls):
        return np.tile(cls._varh_diag, (cls.batch_size,1))

    @classmethod
    def sig2(cls):
        return np.tile(cls._sig2, (cls.batch_size,1))

    @classmethod
    def muh2(cls):
        return np.tile(cls._muh2, (cls.batch_size,1))

    @classmethod
    def varh_diag2(cls):
        return np.tile(cls._varh_diag2, (cls.batch_size,1))
    
    @classmethod
    def wt_TE(cls):
        return np.tile(cls._wt_TE, (cls.batch_size,1,1))

    @classmethod
    def b_TE(cls):
        return np.tile(cls._b_TE, (cls.batch_size,1))

    @classmethod
    def sig2_TE(cls):
        return np.tile(cls._sig2_TE, (cls.batch_size,1))

    @classmethod
    def muh_TE(cls):
        return np.tile(cls._muh_TE, (cls.batch_size,1))

    @classmethod
    def varh_diag_TE(cls):
        return np.tile(cls._varh_diag_TE, (cls.batch_size,1))

    @classmethod
    def muh_TE2(cls):
        return np.tile(cls._muh_TE2, (cls.batch_size,1))

    @classmethod
    def varh_diag_TE2(cls):
        return np.tile(cls._varh_diag_TE2, (cls.batch_size,1))

    @classmethod
    def varh(cls):
        return np.tile(np.diag(cls._varh_diag), (cls.batch_size,1,1))

    @classmethod
    def muh1(cls):
        return cls.muh()

    @classmethod
    def varh_diag1(cls):
        return cls.varh_diag()

    @classmethod
    def varh1(cls):
        return cls.varh()

    @classmethod
    def varh2(cls):
        return cls.varh_diag2()

    @classmethod
    def varh_TE(cls):
        return np.tile(np.diag(cls._varh_diag_TE), (cls.batch_size,1,1))

    @classmethod
    def muh_TE1(cls):
        return cls.muh_TE()

    @classmethod
    def varh_TE1(cls):
        return cls.varh_TE()

    @classmethod
    def varh_TE2(cls):
        return np.tile(np.diag(cls._varh_diag_TE2), (cls.batch_size,1,1))

@tf.keras.utils.register_keras_serializable(package="physDBD")
class SingleLayerModel(tf.keras.Model):

    def __init__(self, lyr, **kwargs):
        super(SingleLayerModel, self).__init__(name='')
        self.lyr = lyr

    def get_config(self):
        config = super(SingleLayerModel, self).get_config()
        config.update({
            "lyr": self.lyr.get_config()
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input_tensor, training=False):
        return self.lyr(input_tensor)

class TestNet:

    def assert_equal_dicts(self, x_out, x_out_true):
        # Convert x_out_true varh to varh_diag as needed
        y_out_true = x_out_true
        for key,val in x_out_true.items():
            print(key)
            if key == "varh":
                y_out_true["varh_diag"] = np.diag(val)
        x_out_true = y_out_true

        for key, val in x_out.items():
            val_true = x_out_true[key]

            self.assert_equal_arrs(val,val_true)

    def assert_equal_arrs(self, x_out, x_out_true):
        tol = 1.e-4
        assert np.max(abs(x_out-x_out_true)) < tol

    def get_random_var(self, batch_size: int, nv: int, nh: int):
        nvar = np.random.rand(batch_size,nv+nh,nv+nh)
        nvar += np.transpose(nvar,axes=[0,2,1])
        return nvar

    def test_fourier(self):

        v = Vals()

        # Create layer
        fl = FourierLatentLayer(
            freqs=v.freqs(),
            offset_fixed=0.0,
            sin_coeff=v.muh_sin_coeffs_init(),
            cos_coeff=v.muh_cos_coeffs_init()
            )

        # Input
        x_in = {
            "tpt": tf.constant(v.tpt(), dtype="float32")
        }
        
        # Output
        x_out = fl(x_in)

        print(x_out)

        x_out_true = np.full(v.batch_size,-0.110302)

        self.assert_equal_arrs(x_out, x_out_true)

        # Create layer
        fl = FourierLatentLayer(
            freqs=v.freqs(),
            offset_fixed=1.01,
            sin_coeff=v.varh_sin_coeffs_init(),
            cos_coeff=v.varh_cos_coeffs_init()
            )

        # Input
        x_in = {
            "tpt": tf.constant(v.tpt(), dtype="float32")
        }
        
        # Output
        x_out = fl(x_in)

        print(x_out)

        x_out_true = np.full(v.batch_size,0.760949)

        self.assert_equal_arrs(x_out, x_out_true)

        # Test save
        model = SingleLayerModel(fl)

        # Call the model once to build it
        x_out = model(x_in)

        # Save
        model.save("model")

        # Test load
        model = tf.keras.models.load_model("model")

    def test_convert_params_layer(self):

        v = Vals()

        lyr = ConvertParamsLayer()

        x_in = {
            "b1": tf.constant(v.b(), dtype="float32"),
            "wt1": tf.constant(v.wt(), dtype="float32"),
            "muh1": tf.constant(v.muh1(), dtype="float32"),
            "muh2": tf.constant(v.muh2(), dtype="float32"),
            "varh_diag1": tf.constant(v.varh_diag1(), dtype="float32"),
            "varh_diag2": tf.constant(v.varh_diag2(), dtype="float32")
            }

        x_out = lyr(x_in)
        
        x_out_true = {
            "wt2": np.array([[4.47214, 8.94427, 17.8885], [1.06066, 3.18198, 3.18198]]),
            "b2": np.array([2.40161, 8.62124, -1.21157])
        }

        self.assert_equal_dicts(x_out, x_out_true)

        print(x_out)

    def test_convert_from_0(self):

        v = Vals()

        lyr = ConvertParamsLayerFrom0()

        x_in = {
            "b1": tf.constant(v.b(), dtype="float32"),
            "wt1": tf.constant(v.wt(), dtype="float32"),
            "muh2": tf.constant(v.muh2(), dtype="float32"),
            "varh_diag2": tf.constant(v.varh_diag2(), dtype="float32")
            }

        x_out = lyr(x_in)
        
        print(x_out)

        x_out_true = {
            "wt2": np.array([[2., 4., 8.], [0.353553, 1.06066, 1.06066]]),
            "b2": np.array([-4.06066, -10.182, -21.182])
        }

        self.assert_equal_dicts(x_out, x_out_true)

    def test_convert_params0_to_params(self):
        
        v = Vals()

        lyr = ConvertParams0ToParamsLayer(
            nv=v.nv,
            nh=v.nh,
            freqs=v.freqs(),
            muh_sin_coeffs_init=v.muh_sin_coeffs_init(),
            muh_cos_coeffs_init=v.muh_cos_coeffs_init(),
            varh_sin_coeffs_init=v.varh_sin_coeffs_init(),
            varh_cos_coeffs_init=v.varh_cos_coeffs_init()
        )

        # Input
        x_in = {
            "tpt": tf.constant(v.tpt(), dtype='float32'),
            "b": tf.constant(v.b(), dtype="float32"),
            "wt": tf.constant(v.wt(), dtype="float32"),
            "sig2": tf.constant(v.sig2(), dtype='float32')
            }   
             
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "b": np.array([3., 5., 6.]), 
            "wt": np.array([[1.99007, 3.98015, 7.9603], [0.995037, 2.98511, 2.98511]]), 
            "muh": np.array([0., 0.]), 
            "varh": np.array([[1.01, 0.], [0., 1.01]]),
            "sig2": np.array([1.])
        }
        
        self.assert_equal_dicts(x_out,x_out_true)
    
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
            "tpt": tf.constant(np.arange(4,4+batch_size), dtype='float32'),
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
        batch_size = 2
        x_in = {
            "mu": tf.constant(np.random.rand(batch_size,nv+nh), dtype="float32"),
            "muTE": tf.constant(np.random.rand(batch_size,nv+nh), dtype="float32"),
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
        batch_size = 2
        x_in = {
            "muTE": tf.constant(np.random.rand(batch_size,nv+nh), dtype="float32"),
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
        batch_size = 2
        x_in = {
            "muvTE": tf.constant(np.random.rand(batch_size,nv), dtype="float32"),
            "varvhTE": tf.constant(np.random.rand(batch_size,nh,nv), dtype="float32"),
            "varh_diagTE": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "varh_diag": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "muh": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "varvh": tf.constant(np.random.rand(batch_size,nh,nv), dtype="float32"),
            "muhTE": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "varvbarTE": tf.constant(np.random.rand(batch_size), dtype="float32")
            }
        
        # Output
        x_out = lyr(x_in)

        print(x_out)

    def test_convert_paramsTE_to_params0TE(self):
        lyr = ConvertParamsTEtoParams0TE()

        nv = 3
        nh = 2
        batch_size = 2
        x_in = {
            "bTE1": tf.constant(np.random.rand(batch_size,nv), dtype="float32"),
            "wtTE1": tf.constant(np.random.rand(batch_size,nh,nv), dtype="float32"),
            "muh1": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "wt1": tf.constant(np.random.rand(batch_size,nh,nv), dtype="float32"),
            "muhTE1": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "varh_diag1": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "varh_diagTE1": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "sig2TE": tf.constant(np.random.rand(batch_size), dtype="float32")
        }
                
        # Output
        x_out = lyr(x_in)

        print(x_out)

    def test_convert_nmomentsTE_to_params0TE(self):

        nv = 3
        nh = 2
        lyr = ConvertNMomentsTEtoParams0TE(nv,nh)

        # Input
        batch_size = 2
        x_in = {
            "mu": tf.constant(np.random.rand(batch_size,nv+nh), dtype="float32"),
            "muTE": tf.constant(np.random.rand(batch_size,nv+nh), dtype="float32"),
            "nvarTE": tf.constant(self.get_random_var(batch_size,nv,nh), dtype="float32"),
            "varh_diag": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "muh": tf.constant(np.random.rand(batch_size,nh), dtype="float32"),
            "var": tf.constant(self.get_random_var(batch_size,nv,nh), dtype="float32"),
            "wt": tf.constant(np.random.rand(batch_size,nh,nv), dtype="float32")
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
        batch_size = 2
        x_in = {
            "tpt": tf.constant(np.arange(3,3+batch_size), dtype='float32'),
            "b": tf.constant(np.random.rand(batch_size,nv), dtype="float32"),
            "wt": tf.constant(np.random.rand(batch_size,nh,nv), dtype="float32"),
            "sig2": tf.constant(np.random.rand(batch_size), dtype='float32')
            }
             
        # Output
        x_out = lyr(x_in)
        
        print(x_out)