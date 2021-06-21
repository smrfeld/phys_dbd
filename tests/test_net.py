from physDBD import  FourierLatentLayer, \
    ConvertParamsLayer, ConvertParamsLayerFrom0, ConvertParams0ToParamsLayer, \
        ConvertParamsToMomentsLayer, ConvertMomentsToNMomentsLayer, DeathRxnLayer, BirthRxnLayer, EatRxnLayer, \
            ConvertNMomentsTEtoMomentsTE, ConvertMomentsTEtoParamMomentsTE, ConvertParamMomentsTEtoParamsTE, \
                ConvertParamsTEtoParams0TE, ConvertNMomentsTEtoParams0TE, ConvertParams0ToNMomentsLayer, RxnInputsLayer

# Depreciation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import copy

import numpy as np
import tensorflow as tf

class Vals:

    nv = 3
    nh = 2
    batch_size = 2
    i_death = 0
    i_birth = 0
    i_predator = 1
    i_prey = 0

    _b = np.array([3.0,5.0,6.0])
    _wt = np.array([[2.0,4.0,8.0],[1.0,3.0,3.0]])
    _muh = np.array([4.0,8.0])
    _varh_diag = np.array([5.0,9.0])
    _sig2 = 1.0
    _varvh = np.array([
        [10., 20., 40.], 
        [9., 27., 27.]
        ])

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

    _mu = np.array([19., 45., 62., 4., 8.])
    _var = np.array([
        [30., 67., 107., 10., 9.], 
        [67., 162., 241., 20., 27.], 
        [107., 241., 402., 40., 27.], 
        [10., 20., 40., 5., 0.], 
        [9., 27., 27., 0., 9.]
        ])
    _nvar = np.array([
        [391., 922., 1285., 86., 161.],
        [922., 2187., 3031., 200., 387.],
        [1285., 3031., 4246., 288., 523.], 
        [86., 200., 288., 21., 32.], 
        [161., 387., 523., 32., 73.]
        ])

    _mu_TE = np.array([3.0, 5.0, 2.0, 1.0, 0.8])
    _nvar_TE = np.array([
        [12.0, 6.0, 3.0, 2.0, 1.0],
        [6.0, 18.0, 4.0, 3.0, 1.0],
        [3.0, 4.0, 16.0, 2.0, 1.0],
        [2.0, 3.0, 2.0, 8.0, 0.5],
        [1.0, 1.0, 1.0, 0.5, 6.0]
    ])
    _var_TE = np.array([
        [-102., -224., -221., -29., -38.2], 
        [-224., -432., -396., -62., -75.], 
        [-221., -396., -232., -68., -64.6], 
        [-29., -62., -68., 0., -10.7], 
        [-38.2, -75., -64.6, -10.7, -6.8]
    ])

    _muv_TE = np.array([3.0,5.0,2.0])
    _varvbar_TE = -766.0
    _varvh_TE = np.array([
        [-29.0, -62.0, -68.0],
        [-38.2, -75.0, -64.60]
    ])

    @classmethod
    def muv_TE(cls):
        return np.tile(cls._muv_TE, (cls.batch_size,1))
    
    @classmethod
    def varvbar_TE(cls):
        return np.full(cls.batch_size, cls._varvbar_TE)

    @classmethod
    def varvh(cls):
        return np.tile(cls._varvh, (cls.batch_size,1,1))

    @classmethod
    def varvh_TE(cls):
        return np.tile(cls._varvh_TE, (cls.batch_size,1,1))

    @classmethod
    def var_TE(cls):
        return np.tile(cls._var_TE, (cls.batch_size,1,1))

    @classmethod
    def mu_TE(cls):
        return np.tile(cls._mu_TE, (cls.batch_size,1))

    @classmethod
    def nvar_TE(cls):
        return np.tile(cls._nvar_TE, (cls.batch_size,1,1))

    @classmethod
    def mu(cls):
        return np.tile(cls._mu, (cls.batch_size,1))

    @classmethod
    def var(cls):
        return np.tile(cls._var, (cls.batch_size,1,1))

    @classmethod
    def nvar(cls):
        return np.tile(cls._nvar, (cls.batch_size,1,1))

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
    def varh_TE(cls):
        return np.tile(np.diag(cls._varh_diag_TE), (cls.batch_size,1))

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
        return {
            "lyr": self.lyr
            }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input_tensor, training=False):
        return self.lyr(input_tensor)

class TestNet:

    def assert_equal_dicts(self, x_out, x_out_true):
        # Convert x_out_true varh to varh_diag as needed
        y_out_true = copy.copy(x_out_true)
        for key,val in x_out_true.items():
            if key == "varh":
                y_out_true["varh_diag"] = np.diag(val)
        if "varh" in y_out_true:
            del y_out_true["varh"]

        for key, val_true in y_out_true.items():
            val = x_out[key]

            self.assert_equal_arrs(val,val_true)

    def assert_equal_arrs(self, x_out, x_out_true):
        tol = 1.e-4
        assert np.max(abs(x_out-x_out_true)) < tol

    def get_random_var(self, batch_size: int, nv: int, nh: int):
        nvar = np.random.rand(batch_size,nv+nh,nv+nh)
        nvar += np.transpose(nvar,axes=[0,2,1])
        return nvar

    def save_load_model(self, lyr, x_in):

        # Test save; call the model once to build it first
        model = SingleLayerModel(lyr)
        x_out = model(x_in)

        print(model)
        model.save("saved_models/model", save_traces=False)

        # Test load
        model_rel = tf.keras.models.load_model("saved_models/model")
        print(model_rel)

        # Check types match!
        # Otherwise we may have: tensorflow.python.keras.saving.saved_model.load.XYZ instead of XYZ
        assert type(model_rel) is type(model)

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

        self.save_load_model(fl, x_in)

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

        self.save_load_model(lyr, x_in)

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

        self.save_load_model(lyr, x_in)

    def test_convert_params0_to_params(self):
        
        v = Vals()

        lyr = ConvertParams0ToParamsLayer.construct(
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
            "b": np.array([3.37934, 5.88512, 7.3909]), 
            "wt": np.array([[2.29273, 4.58545, 9.1709], [1.14636, 3.43909, 3.43909]]), 
            "muh": np.array([-0.110302, -0.110302]), 
            "varh": np.array([[0.760949, 0.], [0., 0.760949]]),
            "sig2": np.array([1.])
        }
        
        self.assert_equal_dicts(x_out,x_out_true)
    
        self.save_load_model(lyr, x_in)

    def test_params_to_moments(self):

        v = Vals()

        lyr = ConvertParamsToMomentsLayer(
            nv=v.nv,
            nh=v.nh
        )

        # Input
        batch_size = 2
        x_in = {
            "b": tf.constant(v.b(), dtype="float32"),
            "wt": tf.constant(v.wt(), dtype="float32"),
            "sig2": tf.constant(v.sig2(), dtype='float32'),
            "varh_diag": tf.constant(v.varh_diag(), dtype='float32'),
            "muh": tf.constant(v.muh(), dtype='float32')
            }   
             
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "mu": np.array([19., 45., 62., 4., 8.]),
            "var": np.array([
                [30., 67., 107., 10., 9.], 
                [67., 162., 241., 20., 27.], 
                [107., 241., 402., 40., 27.], 
                [10., 20., 40., 5., 0.], 
                [9., 27., 27., 0., 9.]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_moments_to_nmoments(self):

        v = Vals()

        lyr = ConvertMomentsToNMomentsLayer()

        # Input
        x_in = {
            "mu": tf.constant(v.mu(), dtype="float32"),
            "var": tf.constant(v.var(), dtype="float32")
            }   
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "mu": np.array([19., 45., 62., 4., 8.]),
            "nvar": np.array([
                [391., 922., 1285., 86., 161.],
                [922., 2187., 3031., 200., 387.],
                [1285., 3031., 4246., 288., 523.], 
                [86., 200., 288., 21., 32.], 
                [161., 387., 523., 32., 73.]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_params0_to_nmoments(self):

        v = Vals()

        lyr = ConvertParams0ToNMomentsLayer.construct(
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
            "mu": np.array([3., 5., 6., -0.110302, -0.110302]),
            "nvar": np.array([
                [15., 26., 37., 1.41374, 0.541419], 
                [26., 51., 71., 2.93779, 2.06546], 
                [37., 71., 110., 6.31678, 1.95516], 
                [1.41374, 2.93779, 6.31678, 0.773116, 0.0121665], 
                [0.541419, 2.06546, 1.95516, 0.0121665, 0.773116]
                ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_death_rxn(self):

        v = Vals()

        lyr = DeathRxnLayer(nv=v.nv,nh=v.nh,i_sp=v.i_death)

        # Input
        x_in = {
            "mu": tf.constant(v.mu(), dtype="float32"),
            "nvar": tf.constant(v.nvar(), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "muTE": np.array([-19., 0., 0., 0., 0.]),
            "nvarTE": np.array([
                [-763., -922., -1285., -86., -161.], 
                [-922., 0., 0., 0., 0.],
                [-1285., 0., 0., 0., 0.], 
                [-86., 0., 0., 0., 0.], 
                [-161., 0., 0., 0., 0.]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_birth_rxn(self):

        v = Vals()

        lyr = BirthRxnLayer(nv=v.nv,nh=v.nh,i_sp=v.i_birth)

        # Input
        x_in = {
            "mu": tf.constant(v.mu(), dtype="float32"),
            "nvar": tf.constant(v.nvar(), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "muTE": np.array([19., 0., 0., 0., 0.]),
            "nvarTE": np.array([
                [801., 922., 1285., 86., 161.], 
                [922., 0., 0., 0., 0.], 
                [1285., 0., 0., 0., 0.], 
                [86., 0., 0., 0., 0.], 
                [161., 0., 0., 0., 0.]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_eat_rxn(self):

        v = Vals()

        lyr = EatRxnLayer(nv=v.nv,nh=v.nh,i_prey=v.i_prey,i_hunter=v.i_predator)

        # Input
        x_in = {
            "mu": tf.constant(v.mu(), dtype="float32"),
            "nvar": tf.constant(v.nvar(), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "muTE": np.array([-922., 922., 0., 0., 0.]),
            "nvarTE": np.array([
                [-39360., -28364., -66558., -4518., -8294.], 
                [-28364., 96088., 66558., 4518., 8294.], 
                [-66558., 66558., 0., 0., 0.], 
                [-4518., 4518., 0., 0., 0.], 
                [-8294., 8294., 0., 0., 0.]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_convert_nmomentsTE_to_momentsTE(self):

        v = Vals()

        lyr = ConvertNMomentsTEtoMomentsTE()

        # Input
        x_in = {
            "mu": tf.constant(v.mu(), dtype="float32"),
            "muTE": tf.constant(v.mu_TE(), dtype="float32"),
            "nvarTE": tf.constant(v.nvar_TE(), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "muTE": np.array([3., 5., 2., 1., 0.8]),
            "varTE": np.array([
                [-102., -224., -221., -29., -38.2], 
                [-224., -432., -396., -62., -75.], 
                [-221., -396., -232., -68., -64.6], 
                [-29., -62., -68., 0., -10.7], 
                [-38.2, -75., -64.6, -10.7, -6.8]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_convert_momentsTE_to_paramMomentsTE(self):

        v = Vals()

        lyr = ConvertMomentsTEtoParamMomentsTE(nv=v.nv,nh=v.nh)

        # Input
        x_in = {
            "muTE": tf.constant(v.mu_TE(), dtype="float32"),
            "varTE": tf.constant(v.var_TE(), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "muhTE": np.array([1.0, 0.8]),
            "muvTE": np.array([3., 5., 2.]),
            "varhTE": np.array([
                [0., -10.7],
                [-10.7, -6.8]
                ]),
            "varvbarTE": -766.,
            "varvhTE": np.array([
                [-29., -62., -68.], 
                [-38.2, -75., -64.6]
                ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_convert_paramMomentsTE_to_paramsTE(self):

        v = Vals()

        lyr = ConvertParamMomentsTEtoParamsTE(nv=v.nv,nh=v.nh)

        # Input
        x_in = {
            "muvTE": tf.constant(v.muv_TE(), dtype="float32"),
            "varvhTE": tf.constant(v.varvh_TE(), dtype="float32"),
            "varhTE": tf.constant(v.varh_TE(), dtype="float32"),
            "varh_diag": tf.constant(v.varh_diag(), dtype="float32"),
            "muh": tf.constant(v.muh(), dtype="float32"),
            "varvh": tf.constant(v.varvh(), dtype="float32"),
            "muhTE": tf.constant(v.muh_TE(), dtype="float32"),
            "varvbarTE": tf.constant(v.varvbar_TE(), dtype="float32")
            }
        
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "bTE": np.array([60.81777, 122.41333, 116.64888]),
            "muhTE": np.array([0.3, 0.8]),
            "varhTE": np.array([
                [0.9, 0],
                [0, 0.7],
                ]),
            "sig2TE": 645.63333,
            "wtTE": np.array([
                [-6.16, -13.12, -15.04], 
                [-4.32222, -8.56667, -7.41111]
                ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_convert_paramsTE_to_params0TE(self):

        v = Vals()

        lyr = ConvertParamsTEtoParams0TE()

        x_in = {
            "bTE1": tf.constant(v.b_TE(), dtype="float32"),
            "wtTE1": tf.constant(v.wt_TE(), dtype="float32"),
            "muh1": tf.constant(v.muh(), dtype="float32"),
            "wt1": tf.constant(v.wt(), dtype="float32"),
            "muhTE1": tf.constant(v.muh_TE(), dtype="float32"),
            "varh_diag1": tf.constant(v.varh_diag(), dtype="float32"),
            "varhTE1": tf.constant(v.varh_TE(), dtype="float32"),
            "sig2TE": tf.constant(v.sig2_TE(), dtype="float32")
        }
                
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "bTE2": np.array([29.7, 36., 41.6]),
            "sig2TE": 0.3,
            "wtTE2": np.array([
                [2.63856, 5.27712, 8.31817],
                [9.11667, 9.35, 9.35]
                ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_convert_nmomentsTE_to_params0TE(self):

        v = Vals()

        lyr = ConvertNMomentsTEtoParams0TE(v.nv,v.nh)

        # Input
        x_in = {
            "mu": tf.constant(v.mu(), dtype="float32"),
            "muTE": tf.constant(v.mu_TE(), dtype="float32"),
            "nvarTE": tf.constant(v.nvar_TE(), dtype="float32"),
            "varh_diag": tf.constant(v.varh_diag(), dtype="float32"),
            "muh": tf.constant(v.muh(), dtype="float32"),
            "var": tf.constant(v.var(), dtype="float32"),
            "wt": tf.constant(v.wt(), dtype="float32")
            }

        print(x_in)
        
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "bTE": np.array([3., 5., 2.00002]),
            "sig2TE": 301.86667,
            "wtTE": np.array([
                [-10.5766, -20.5495, -23.2327],
                [-8.03333, -14.4667, -3.86667]
                ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_rxn_inputs(self):

        v = Vals()

        rxn_specs = [
            ("BIRTH",v.i_birth),
            ("DEATH",v.i_death),
            ("EAT",v.i_predator,v.i_prey)
        ]

        lyr = RxnInputsLayer.construct(
            nv=v.nv,
            nh=v.nh,
            freqs=v.freqs(),
            muh_sin_coeffs_init=v.muh_sin_coeffs_init(),
            muh_cos_coeffs_init=v.muh_cos_coeffs_init(),
            varh_sin_coeffs_init=v.varh_sin_coeffs_init(),
            varh_cos_coeffs_init=v.varh_cos_coeffs_init(),
            rxn_specs=rxn_specs
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

        self.save_load_model(lyr, x_in)