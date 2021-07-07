from physDBD.gauss import FourierLatentGaussLayer, \
    ConvertParamsGaussLayer, ConvertParamsGaussLayerFrom0, \
        ConvertParams0ToParamsGaussLayer, ConvertParamsToMomentsGaussLayer, \
            ConvertMomentsToNMomentsGaussLayer, ConvertParams0ToNMomentsGaussLayer

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

    _mu = np.array([10.0,8.0,4.0,20.0,3.0])

    _mu_h2 = np.array([4.0,80.0])

    _chol = np.array([
        [3.0, 0.0, 0.0, 0.0, 0.0],
        [5.0, 8.0, 0.0, 0.0, 0.0],
        [-3.0, 4.0, 7.0, 0.0, 0.0],
        [-1.0, -3.0, -4.0, 9.0, 0.0],
        [-3.0, 8.0, 10.0, 9.0, 20.0]
        ])

    _chol_vh2 = np.array([
        [-4.0, -5.0, -8.0],
        [-2.0, 9.0, 8.0]
        ])

    _chol_h2 = np.array([
        [15.0, 0.0],
        [12.0, 30.0]
        ])

    _freqs = np.array([1.,2.,3.])
    _cos_coeffs_init = np.array([1.,2.,4.])
    _sin_coeffs_init = np.array([1.,5.,4.])

    _muh_cos_coeffs_init = np.array([2.,5.,3.])
    _muh_sin_coeffs_init = np.array([3.,6.,1.])
    _cholvh_cos_coeffs_init = np.array([1.,8.,4.])
    _cholvh_sin_coeffs_init = np.array([4.,5.,4.])
    _cholh_cos_coeffs_init = np.array([8.,10.,9.])
    _cholh_sin_coeffs_init = np.array([3.,7.,8.])

    @classmethod
    def tpt(cls):
        return np.full(cls.batch_size,3-1) # zero indexed
    
    @classmethod
    def tile_vec(cls, vec: np.array) -> np.array:
        return np.tile(vec, (cls.batch_size,1))

    @classmethod
    def tile_mat(cls, mat: np.array) -> np.array:
        return np.tile(mat, (cls.batch_size,1,1))

    @classmethod
    def mu_v1(cls):
        mu_v1 = cls._mu[:cls.nv]
        return cls.tile_vec(mu_v1)

    @classmethod
    def mu(cls):
        return cls.tile_vec(cls._mu)
    
    @classmethod
    def mu_h2(cls):
        return cls.tile_vec(cls._mu_h2)
    
    @classmethod
    def chol(cls):
        return cls.tile_mat(cls._chol)

    @classmethod
    def chol_v1(cls):
        chol_v1 = cls._chol[:cls.nv,:cls.nv]
        return cls.tile_mat(chol_v1)

    @classmethod
    def chol_vh2(cls):
        return cls.tile_mat(cls._chol_vh2)

    @classmethod
    def chol_h2(cls):
        return cls.tile_mat(cls._chol_h2)
    
    @classmethod
    def cov(cls):
        prec = np.dot(cls._chol, np.transpose(cls._chol))
        cov = np.linalg.inv(prec)
        return cls.tile_mat(cov)
    
    @classmethod
    def cov_v(cls):
        prec = np.dot(cls._chol, np.transpose(cls._chol))
        cov = np.linalg.inv(prec)
        cov_v = cov[:cls.nv,:cls.nv]
        return cls.tile_mat(cov_v)

    @classmethod
    def freqs(cls):
        return cls._freqs

    @classmethod
    def cos_coeffs_init(cls):
        return cls._cos_coeffs_init

    @classmethod
    def sin_coeffs_init(cls):
        return cls._sin_coeffs_init

    @classmethod
    def muh_cos_coeffs_init(cls):
        return cls._muh_cos_coeffs_init

    @classmethod
    def muh_sin_coeffs_init(cls):
        return cls._muh_sin_coeffs_init

    @classmethod
    def cholh_cos_coeffs_init(cls):
        return cls._cholh_cos_coeffs_init

    @classmethod
    def cholh_sin_coeffs_init(cls):
        return cls._cholh_sin_coeffs_init

    @classmethod
    def cholvh_cos_coeffs_init(cls):
        return cls._cholvh_cos_coeffs_init

    @classmethod
    def cholvh_sin_coeffs_init(cls):
        return cls._cholvh_sin_coeffs_init

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

class TestNetGauss:

    def assert_equal_dicts(self, x_out, x_out_true):
        for key, val_true in x_out_true.items():
            val = x_out[key]

            self.assert_equal_arrs(val,val_true)

    def assert_equal_arrs(self, x_out, x_out_true):
        tol = 5.e-4
        assert np.max(abs(x_out-x_out_true)) < tol

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
        fl = FourierLatentGaussLayer(
            freqs=v.freqs(),
            offset=0.0,
            sin_coeff=v.sin_coeffs_init(),
            cos_coeff=v.cos_coeffs_init()
            )

        print("Freqs: ", v.freqs())
        print("Sin: ", v.sin_coeffs_init())
        print("Cos: ", v.cos_coeffs_init())

        # Input
        x_in = {
            "tpt": tf.constant(v.tpt(), dtype="float32")
        }

        print("Inputs: ", x_in)
        
        # Output
        x_out = fl(x_in)

        print("Outputs: ",x_out)

        x_out_true = np.full(v.batch_size,-1.8751299)

        self.assert_equal_arrs(x_out, x_out_true)

        self.save_load_model(fl, x_in)

    def test_convert_params_layer(self):

        v = Vals()

        lyr = ConvertParamsGaussLayer(nv=v.nv,nh=v.nh)

        x_in = {
            "mu1": tf.constant(v.mu(), dtype="float32"),
            "chol1": tf.constant(v.chol(), dtype="float32"),
            "chol_vh2": tf.constant(v.chol_vh2(), dtype="float32"),
            "chol_h2": tf.constant(v.chol_h2(), dtype="float32"),
            "mu_h2": tf.constant(v.mu_h2(), dtype="float32")
            }
        
        print("Inputs: ", x_in)

        x_out = lyr(x_in)
        
        print("Outputs: ",x_out)

        x_out_true = {
            "mu2": np.array([10., 8., 4., 4., 80.]), 
            "chol2": np.array([
                [3.0437, 0., 0., 0, 0], 
                [5.57349, 7.85566, 0., 0, 0],
                [-2.13828, 3.44518, 6.63218, 0, 0], 
                [-4., -5., -8., 15., 0.], 
                [-2., 9., 8., 12., 30.]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_convert_from_0(self):

        v = Vals()

        lyr = ConvertParamsGaussLayerFrom0()

        x_in = {
            "mu_v1": tf.constant(v.mu_v1(), dtype="float32"),
            "chol_v1": tf.constant(v.chol_v1(), dtype="float32"),
            "mu_h2": tf.constant(v.mu_h2(), dtype="float32"),
            "chol_vh2": tf.constant(v.chol_vh2(), dtype="float32"),
            "chol_h2": tf.constant(v.chol_h2(), dtype="float32")
            }

        print("Input: ", x_in)

        x_out = lyr(x_in)
        
        print("Output: ", x_out)

        x_out_true = {
            "mu2": np.array([10., 8., 4., 4., 80.]),
            "chol2": np.array([
                [3.07698, 0., 0., 0, 0], 
                [5.6037, 8.76592, 0., 0, 0], 
                [-1.92121, 6.57703, 8.61553, 0, 0], 
                [-4., -5., -8., 15., 0.], 
                [-2., 9., 8., 12., 30.]
                ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_convert_params0_to_params(self):
        
        v = Vals()

        lyr = ConvertParams0ToParamsGaussLayer.construct(
            nv=v.nv,
            nh=v.nh,
            freqs=v.freqs(),
            muh_sin_coeffs_init=v.muh_sin_coeffs_init(),
            muh_cos_coeffs_init=v.muh_cos_coeffs_init(),
            cholh_cos_coeffs_init=v.cholh_cos_coeffs_init(),
            cholh_sin_coeffs_init=v.cholh_sin_coeffs_init(),
            cholvh_cos_coeffs_init=v.cholvh_cos_coeffs_init(),
            cholvh_sin_coeffs_init=v.cholvh_sin_coeffs_init()
        )

        # Input
        x_in = {
            "tpt": tf.constant(v.tpt(), dtype='float32'),
            "mu_v": tf.constant(v.mu_v1(), dtype="float32"),
            "chol_v": tf.constant(v.chol_v1(), dtype="float32")
            }   
        
        print("Inputs: ", x_in)

        # Output
        x_out = lyr(x_in)

        print("Outputs: ", x_out)

        x_out_true = {
            "mu": np.array([10., 8., 4., -3.31234, -3.31234]), 
            "chol": np.array([
                [3.24593, 0., 0., 0, 0], 
                [6.90921, 8.78471, 0., 0, 0], 
                [-0.879767, 6.00886, 7.85476, 0, 0], 
                [-3.0691, -3.0691, -3.0691, -6.02913, 0], 
                [-3.0691, -3.0691, -3.0691, -6.02913, -6.02913]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)
    
        self.save_load_model(lyr, x_in)

    def test_params_to_moments(self):

        v = Vals()

        lyr = ConvertParamsToMomentsGaussLayer(
            nv=v.nv,
            nh=v.nh
        )

        # Input
        x_in = {
            "mu": tf.constant(v.mu(), dtype="float32"),
            "chol": tf.constant(v.chol(), dtype="float32")
            }   
             
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "mu": np.array([10., 8., 4., 20., 3.]), 
            "cov": np.array([
                [0.231417, -0.0432518, 0.0462896, 0.0111036, -0.00177083], 
                [-0.0432518, 0.021177, -0.0076992, 0.00203979, -0.0009375], 
                [0.0462896, -0.0076992, 0.0344394, 0.0120547, -0.005], 
                [0.0111036, 0.00203979, 0.0120547, 0.0148457, -0.0025], 
                [-0.00177083, -0.0009375, -0.005, -0.0025, 0.0025]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_moments_to_nmoments(self):

        v = Vals()

        lyr = ConvertMomentsToNMomentsGaussLayer()

        # Input
        x_in = {
            "mu": tf.constant(v.mu(), dtype="float32"),
            "cov": tf.constant(v.cov(), dtype="float32")
            }   
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "mu": np.array([10., 8., 4., 20., 3.]),
            "ncov": np.array([
                [100.231, 79.9567, 40.0463, 200.011, 29.9982],
                [79.9567, 64.0212, 31.9923, 160.002, 23.9991],
                [40.0463, 31.9923, 16.0344, 80.0121, 11.995], 
                [200.011, 160.002, 80.0121, 400.015, 59.9975], 
                [29.9982, 23.9991, 11.995, 59.9975, 9.0025]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_params0_to_nmoments(self):

        v = Vals()

        lyr = ConvertParams0ToNMomentsGaussLayer.construct(
            nv=v.nv,
            nh=v.nh,
            freqs=v.freqs(),
            muh_sin_coeffs_init=v.muh_sin_coeffs_init(),
            muh_cos_coeffs_init=v.muh_cos_coeffs_init(),
            cholvh_sin_coeffs_init=v.cholvh_sin_coeffs_init(),
            cholvh_cos_coeffs_init=v.cholvh_cos_coeffs_init(),
            cholh_sin_coeffs_init=v.cholh_sin_coeffs_init(),
            cholh_cos_coeffs_init=v.cholh_cos_coeffs_init()
        )

        # Input
        x_in = {
            "tpt": tf.constant(v.tpt(), dtype='float32'),
            "mu_v": tf.constant(v.mu_v1(), dtype="float32"),
            "chol_v": tf.constant(v.chol_v1(), dtype="float32")
            }
             
        # Output
        x_out = lyr(x_in)
        
        print(x_out)
    
        x_out_true = {
            "mu": np.array([10., 8., 4., -3.31234, -3.31234]),
            "ncov": np.array([
                [100.223, 79.9553, 40.0374, -33.0993, -33.1234], 
                [79.9553, 64.0207, 31.9898, -26.4965, -26.4987], 
                [40.0374, 31.9898, 16.0204, -13.2386, -13.2494], 
                [-33.0993, -26.4965, -13.2386, 11.0266, 10.9441], 
                [-33.1234, -26.4987, -13.2494, 10.9441, 10.9991]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    '''
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
    '''