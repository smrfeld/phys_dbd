from helpers_test import SingleLayerModel
from physDBD.gauss import FourierLatentGaussLayer, \
    ConvertParamsGaussLayer, ConvertParamsGaussLayerFrom0, \
        ConvertParams0ToParamsGaussLayer, ConvertParamsToMomentsGaussLayer, \
            ConvertParams0ToNMomentsGaussLayer

# Depreciation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import copy

import numpy as np
import tensorflow as tf

def tile_vec(vec: np.array, batch_size: int) -> np.array:
    return np.tile(vec, (batch_size,1))

def tile_mat(mat: np.array, batch_size: int) -> np.array:
    return np.tile(mat, (batch_size,1,1))

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

        freqs = np.array([1.,2.,3.])
        cos_coeffs_init = np.array([1.,2.,4.])
        sin_coeffs_init = np.array([1.,5.,4.])

        batch_size = 2
        tpt = np.full(batch_size,3-1) # zero indexed

        # Create layer
        fl = FourierLatentGaussLayer(
            freqs=freqs,
            offset=0.0,
            sin_coeff=sin_coeffs_init,
            cos_coeff=cos_coeffs_init
            )

        print("Freqs: ", freqs)
        print("Sin: ", sin_coeffs_init)
        print("Cos: ", cos_coeffs_init)

        # Input
        x_in = {
            "tpt": tf.constant(tpt, dtype="float32")
        }

        print("Inputs: ", x_in)
        
        # Output
        x_out = fl(x_in)

        print("Outputs: ",x_out)

        x_out_true = np.full(batch_size,-1.8751299)

        self.assert_equal_arrs(x_out, x_out_true)

        self.save_load_model(fl, x_in)

    def test_convert_params_layer(self):

        nv = 3
        nh = 2
        batch_size = 2

        mu1 = np.array([10,3,4,6,8])
        mu1 = tile_vec(mu1,batch_size)

        chol1 = np.array([
            [30,0,0,0,0],
            [20,11,0,0,0],
            [4,23,14,0,0],
            [5,8,3,29,0],
            [34,5,77,1,34]
            ])
        chol1 = tile_mat(chol1,batch_size)

        chol_hv2 = np.array([
            [5,3,7],
            [1,5,7]
            ])
        chol_hv2 = tile_mat(chol_hv2,batch_size)

        chol_h2 = np.array([
            [3,0],
            [4,2]
            ])
        chol_h2 = tile_mat(chol_h2,batch_size)

        mu_h2 = np.array([7,32])
        mu_h2 = tile_vec(mu_h2,batch_size)

        lyr = ConvertParamsGaussLayer(nv=nv,nh=nh)

        x_in = {
            "mu1": tf.constant(mu1, dtype="float32"),
            "chol1": tf.constant(chol1, dtype="float32"),
            "chol_hv2": tf.constant(chol_hv2, dtype="float32"),
            "chol_h2": tf.constant(chol_h2, dtype="float32"),
            "mu_h2": tf.constant(mu_h2, dtype="float32")
            }
        
        print("Inputs: ", x_in)

        x_out = lyr(x_in)
        
        print("Outputs: ",x_out)

        x_out_true = {
            "mu2": np.array([10., 3., 4., 7., 32.]), 
            "chol2": np.array([
                [56.505, 0., 0., 0, 0], 
                [25.9328, 14.4518, 0., 0, 0],
                [-11.3685, 32.7313, 15.8033, 0, 0], 
                [5., 3., 7., 3., 0.], 
                [1., 5., 7., 4., 2.]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_convert_from_0(self):

        batch_size = 2

        mu_v1 = np.array([10,3,4])
        mu_v1 = tile_vec(mu_v1,batch_size)

        chol_v1 = np.array([
            [10,0,0],
            [4,6,0],
            [8,4,34]
            ])
        chol_v1 = tile_mat(chol_v1,batch_size)

        mu_h2 = np.array([35,77])
        mu_h2 = tile_vec(mu_h2,batch_size)

        chol_hv2 = np.array([
            [135,46,32],
            [42,1,34]
            ])
        chol_hv2 = tile_mat(chol_hv2,batch_size)

        chol_h2 = np.array([
            [15,0],
            [4,35]
            ])
        chol_h2 = tile_mat(chol_h2,batch_size)

        lyr = ConvertParamsGaussLayerFrom0()

        x_in = {
            "mu_v1": tf.constant(mu_v1, dtype="float32"),
            "chol_v1": tf.constant(chol_v1, dtype="float32"),
            "mu_h2": tf.constant(mu_h2, dtype="float32"),
            "chol_hv2": tf.constant(chol_hv2, dtype="float32"),
            "chol_h2": tf.constant(chol_h2, dtype="float32")
            }

        print("Input: ", x_in)

        x_out = lyr(x_in)
        
        print("Output: ", x_out)

        x_out_true = {
            "mu2": np.array([10, 3, 4, 35, 77]),
            "chol2": np.array([
                [25.3647, 0., 0., 0, 0], 
                [32.7391, 11.946045, 0., 0, 0], 
                [301.80923, 94.94065, 83.8398, 0, 0], 
                [135., 46., 32., 15., 0.], 
                [42., 1., 34., 4., 35.]
                ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_convert_params0_to_params(self):
        nv = 3
        nh = 2
        freqs = np.array([1.,2.,3.])
        muh_cos_coeffs_init = np.array([2.,5.,3.])
        muh_sin_coeffs_init = np.array([3.,6.,1.])
        cholhv_cos_coeffs_init = np.array([1.,8.,4.])
        cholhv_sin_coeffs_init = np.array([4.,5.,4.])
        cholh_cos_coeffs_init = np.array([8.,10.,9.])
        cholh_sin_coeffs_init = np.array([3.,7.,8.])

        batch_size = 2

        tpt = np.full(batch_size,3-1) # zero indexed

        mu_v1 = np.array([10.0,8.0,4.0])
        mu_v1 = tile_vec(mu_v1,batch_size)

        chol_v1_non_zero = np.array([3.0, 5.0, 8.0, 4.0, 7.0])
        chol_v1_non_zero = tile_vec(chol_v1_non_zero,batch_size)

        non_zero_idx_pairs_vv = [(0,0),(1,0),(1,1),(2,1),(2,2)]
        non_zero_idx_pairs_hv = [(0,0),(0,1),(0,2),(1,1),(1,2)]
        non_zero_idx_pairs_hh = [(0,0),(1,0),(1,1)]

        lyr = ConvertParams0ToParamsGaussLayer.construct(
            nv=nv,
            nh=nh,
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            cholh_cos_coeffs_init=cholh_cos_coeffs_init,
            cholh_sin_coeffs_init=cholh_sin_coeffs_init,
            cholhv_cos_coeffs_init=cholhv_cos_coeffs_init,
            cholhv_sin_coeffs_init=cholhv_sin_coeffs_init,
            non_zero_idx_pairs_vv=non_zero_idx_pairs_vv,
            non_zero_idx_pairs_hv=non_zero_idx_pairs_hv,
            non_zero_idx_pairs_hh=non_zero_idx_pairs_hh
        )

        # Input
        x_in = {
            "tpt": tf.constant(tpt, dtype='float32'),
            "mu_v": tf.constant(mu_v1, dtype="float32"),
            "chol_v_non_zero": tf.constant(chol_v1_non_zero, dtype="float32")
            }   
        
        print("Inputs: ", x_in)

        # Output
        x_out = lyr(x_in)

        print("Outputs: ", x_out)

        x_out_true = {
            "mu": np.array([10., 8., 4., -3.31234, -3.31234]), 
            "chol": np.array([
                [ 3.5872293,  0.       ,  0.       ,  0.       ,  0.       ],
                [ 7.4780407,  8.784711 ,  0.       ,  0.       ,  0.       ],
                [ 2.3661642,  6.0088573,  7.8547583,  0.       ,  0.       ],
                [-3.0690994, -3.0690994, -3.0690994, -6.029127 ,  0.       ],
                [ 0.       , -3.0690994, -3.0690994, -6.029127 , -6.029127 ]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)
    
        self.save_load_model(lyr, x_in)

    def test_params_to_moments(self):

        nv = 3
        nh = 2
        batch_size = 3

        mu = np.array([23,53,34,66,12])
        mu = tile_vec(mu,batch_size)

        chol = np.array([
            [30,0,0,0,0],
            [12,24,0,0,0],
            [23,43,34,0,0],
            [12,14,18,24,0],
            [34,56,88,32,11]
        ])
        chol = tile_mat(chol,batch_size)

        lyr = ConvertParamsToMomentsGaussLayer(
            nv=nv,
            nh=nh
        )

        # Input
        x_in = {
            "mu": tf.constant(mu, dtype="float32"),
            "chol": tf.constant(chol, dtype="float32")
            }   
             
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "mu": np.array([23., 53., 34., 66., 12.]), 
            "cov": np.array([
                [0.00153703, 0.00104118, -0.00172425, -0.00152538, 0.000961487], 
                [0.00104118, 0.0320294, -0.0301518, -0.0193879, 0.0150165], 
                [-0.00172425, -0.0301518, 0.0306347, 0.0198231, -0.0155566], 
                [-0.00152538, -0.0193879, 0.0198231, 0.0164285, -0.0110193], 
                [0.000961487, 0.0150165, -0.0155566, -0.0110193, 0.00826446]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_params0_to_nmoments(self):

        nv = 3
        nh = 2

        freqs = np.array([1.,2.,3.])
        muh_cos_coeffs_init = np.array([2.,5.,3.])
        muh_sin_coeffs_init = np.array([3.,6.,1.])
        cholhv_cos_coeffs_init = np.array([1.,8.,4.])
        cholhv_sin_coeffs_init = np.array([4.,5.,4.])
        cholh_cos_coeffs_init = np.array([8.,10.,9.])
        cholh_sin_coeffs_init = np.array([3.,7.,8.])

        batch_size = 2

        non_zero_idx_pairs_vv = [(0,0),(1,0),(1,1),(2,1),(2,2)]
        non_zero_idx_pairs_hv = [(0,0),(0,1),(0,2),(1,1),(1,2)]
        non_zero_idx_pairs_hh = [(0,0),(1,0),(1,1)]
        
        tpt = np.full(batch_size,3-1) # zero indexed

        mu_v1 = np.array([10.0,8.0,4.0])
        mu_v1 = tile_vec(mu_v1,batch_size)

        chol_v1_non_zero = np.array([3.0, 5.0, 8.0, 4.0, 7.0])
        chol_v1_non_zero = tile_vec(chol_v1_non_zero,batch_size)

        lyr = ConvertParams0ToNMomentsGaussLayer.construct(
            nv=nv,
            nh=nh,
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            cholhv_sin_coeffs_init=cholhv_sin_coeffs_init,
            cholhv_cos_coeffs_init=cholhv_cos_coeffs_init,
            cholh_sin_coeffs_init=cholh_sin_coeffs_init,
            cholh_cos_coeffs_init=cholh_cos_coeffs_init,
            non_zero_idx_pairs_vv=non_zero_idx_pairs_vv,
            non_zero_idx_pairs_hv=non_zero_idx_pairs_hv,
            non_zero_idx_pairs_hh=non_zero_idx_pairs_hh
        )

        # Input
        x_in = {
            "tpt": tf.constant(tpt, dtype='float32'),
            "mu_v": tf.constant(mu_v1, dtype="float32"),
            "chol_v_non_zero": tf.constant(chol_v1_non_zero, dtype="float32")
            }
             
        # Output
        x_out = lyr(x_in)
        
        print(x_out)
    
        x_out_true = {
            "mu": np.array([10., 8., 4., -3.31234, -3.31234]),
            "ncov": np.array([
                [100.168686,  79.965454,  40.017006, -33.088116, -33.146927],
                [ 79.965454,  64.02073 ,  31.989796, -26.496452, -26.49871 ],
                [ 40.017006,  31.989796,  16.020409, -13.238606, -13.249355],
                [-33.088116, -26.496452, -13.238606,  11.026608,  10.944078],
                [-33.146927, -26.49871 , -13.249355,  10.944078,  10.999098]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    '''
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