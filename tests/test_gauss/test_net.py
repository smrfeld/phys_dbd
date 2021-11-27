from helpers_test import SingleLayerModel
from physDBD.gauss import FourierLatentGaussLayer, \
    ConvertParamsGaussLayer, ConvertParamsGaussLayerFrom0, \
        ConvertParams0ToParamsGaussLayer, ConvertParamsToMomentsGaussLayer, \
            ConvertParams0ToNMomentsGaussLayer, ConvertMomentsTEtoParamsTEGaussLayer, \
                ConvertParamsTEtoParams0TEGaussLayer, ConvertMomentsTEtoParams0TEGaussLayer, \
                    RxnInputsGaussLayer

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
        batch_size = 2

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

        chol_v1_non_zero = np.array([3.0, 5.0, 8.0, -3.0, 4.0])
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
                [100.179, 79.9593, 39.9609, -33.1088, -33.1469], 
                [79.9593, 64.0244, 32.0234, -26.484, -26.4987], 
                [39.9609, 32.0234, 16.0625, -13.2305, -13.2494], 
                [-33.1088, -26.484, -13.2305, 11.0266, 10.9441], 
                [-33.1469, -26.4987, -13.2494, 10.9441, 10.9991]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_convert_momentsTE_to_paramsTE(self):

        nv = 3
        nh = 2
        batch_size = 2

        chol = np.array([
            [3,0,0,0,0],
            [1,2,0,0,0],
            [2,4,3,0,0],
            [1,1,1,2,0],
            [3,5,8,3,1]
        ])
        chol = tile_mat(chol,batch_size)

        muTE = np.array([4,2,3,3,1])
        muTE = tile_vec(muTE,batch_size)

        covTE = np.array([
            [4,6,2,8,9],
            [6,7,4,3,9],
            [2,4,6,8,9],
            [8,3,8,2,6],
            [9,9,9,6,3]
        ])
        covTE = tile_mat(covTE,batch_size)

        lyr = ConvertMomentsTEtoParamsTEGaussLayer(nv=nv,nh=nh)

        # Input
        x_in = {
            "muTE": tf.constant(muTE, dtype="float32"),
            "covTE": tf.constant(covTE, dtype="float32"),
            "chol": tf.constant(chol, dtype="float32")
            }
        
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "muTE": np.array([4,2,3,3,1]),
            "cholTE": np.array([
                [-927., 0., 0., 0., 0.],
                [-1833., -941., 0., 0., 0.],
                [-6024., -4657., -1236., 0., 0.],
                [-2523., -2125.5, -1050., -107., 0.],
                [-12093., -10922.5, -4310., -181.5, -1.5]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_convert_momentsTE_to_paramsTE(self):

        nv = 3
        nh = 2
        batch_size = 2

        mu_TE = np.array([3,5,7,3,2])

        chol = np.array([
            [3,0,0,0,0],
            [1,2,0,0,0],
            [2,4,3,0,0],
            [1,1,1,2,0],
            [3,5,8,3,1]
            ])

        chol_TE = np.array([
            [2,0,0,0,0],
            [3,4,0,0,0],
            [1,2,3,0,0],
            [5,7,3,9,0],
            [2,3,5,4,8]
            ])

        prec = np.dot(chol,np.transpose(chol))
        prec_h = prec[nv:,nv:]
        prec_h_inv = np.linalg.inv(prec_h)

        chol_v = chol[:nv,:nv]
        chol_h = chol[nv:,nv:]
        chol_hv = chol[nv:,:nv]

        chol_v_TE = chol_TE[:nv,:nv]
        chol_h_TE = chol_TE[nv:,nv:]
        chol_hv_TE = chol_TE[nv:,:nv]

        amat = np.eye(nv) - np.dot(np.transpose(chol_hv),np.dot(prec_h_inv,chol_hv))
        chol_amat = np.linalg.cholesky(amat)

        lyr = ConvertParamsTEtoParams0TEGaussLayer(nv=nv,nh=nh)

        # Tile inputs
        mu_TE = tile_vec(mu_TE,batch_size)
        prec_h = tile_mat(prec_h,batch_size)
        chol_v = tile_mat(chol_v,batch_size)
        chol_h = tile_mat(chol_h,batch_size)
        chol_hv = tile_mat(chol_hv,batch_size)
        chol_v_TE = tile_mat(chol_v_TE,batch_size)
        chol_h_TE = tile_mat(chol_h_TE,batch_size)
        chol_hv_TE = tile_mat(chol_hv_TE,batch_size)
        chol_amat = tile_mat(chol_amat,batch_size)

        # Input
        x_in = {
            "mu_TE": tf.constant(mu_TE, dtype="float32"),
            "prec_h": tf.constant(prec_h, dtype="float32"),
            "chol_v": tf.constant(chol_v, dtype="float32"),
            "chol_h": tf.constant(chol_h, dtype="float32"),
            "chol_hv": tf.constant(chol_hv, dtype="float32"),
            "chol_v_TE": tf.constant(chol_v_TE, dtype="float32"),
            "chol_h_TE": tf.constant(chol_h_TE, dtype="float32"),
            "chol_hv_TE": tf.constant(chol_hv_TE, dtype="float32"),
            "chol_amat": tf.constant(chol_amat, dtype="float32")
            }

        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "muv_TE": np.array([3,5,7]),
            "cholv_TE_std": np.array([
                [1.70733, 0., 0.],
                [1.36787, 3.44805, 0.], 
                [-0.336917, 2.82135, 3.63626]
                ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_convert_momentsTE_to_params0TE(self):

        nv = 3
        nh = 2
        batch_size = 2

        chol = np.array([
            [3,0,0,0,0],
            [1,2,0,0,0],
            [2,4,3,0,0],
            [1,1,1,2,0],
            [3,5,8,3,1]
        ])
        prec = np.dot(chol,np.transpose(chol))
        prec_h = prec[nv:,nv:]
        prec_h_inv = np.linalg.inv(prec_h)

        muTE = np.array([4,2,3,3,1])

        covTE = np.array([
            [4,6,2,8,9],
            [6,7,4,3,9],
            [2,4,6,8,9],
            [8,3,8,2,6],
            [9,9,9,6,3]
        ])

        chol_hv = chol[nv:,:nv]
        amat = np.eye(nv) - np.dot(np.transpose(chol_hv),np.dot(prec_h_inv,chol_hv))
        chol_amat = np.linalg.cholesky(amat)

        chol = tile_mat(chol,batch_size)
        prec_h = tile_mat(prec_h,batch_size)
        muTE = tile_vec(muTE,batch_size)
        covTE = tile_mat(covTE,batch_size)
        chol_amat = tile_mat(chol_amat,batch_size)

        lyr = ConvertMomentsTEtoParams0TEGaussLayer(nv=nv, nh=nh)

        # Input
        x_in = {
            "prec_h": tf.constant(prec_h, dtype="float32"),
            "chol_amat": tf.constant(chol_amat, dtype="float32"),
            "muTE": tf.constant(muTE, dtype="float32"),
            "covTE": tf.constant(covTE, dtype="float32"),
            "chol": tf.constant(chol, dtype="float32")
            }

        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "muv_TE": np.array([4,2,3]),
            "cholv_TE_std": np.array([
                [-101.285706, 0., 0.], 
                [-143.94946, -65.553345, 0.], 
                [-179.96094, -83.13623, -0.28163147]
                ])
            }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)

    def test_rxn_inputs(self):

        nv = 3
        nh = 2

        i_birth = 0
        i_death = 1
        i_predator = 1
        i_prey = 0

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
        
        mu_v1 = np.array([10,3,4])
        mu_v1 = tile_vec(mu_v1,batch_size)

        chol_v1_non_zero = np.array([3.0, 5.0, 8.0, 4.0, 7.0])
        chol_v1_non_zero = tile_vec(chol_v1_non_zero,batch_size)

        tpt = np.full(batch_size,3-1) # zero indexed

        rxn_specs = [
            ("BIRTH",i_birth),
            ("DEATH",i_death),
            ("EAT",i_predator,i_prey)
        ]

        lyr = RxnInputsGaussLayer.construct(
            nv=nv,
            nh=nh,
            freqs=freqs,
            muh_sin_coeffs_init=muh_sin_coeffs_init,
            muh_cos_coeffs_init=muh_cos_coeffs_init,
            cholhv_cos_coeffs_init=cholhv_cos_coeffs_init,
            cholhv_sin_coeffs_init=cholhv_sin_coeffs_init,
            cholh_cos_coeffs_init=cholh_cos_coeffs_init,
            cholh_sin_coeffs_init=cholh_sin_coeffs_init,
            non_zero_idx_pairs_vv=non_zero_idx_pairs_vv,
            non_zero_idx_pairs_hv=non_zero_idx_pairs_hv,
            non_zero_idx_pairs_hh=non_zero_idx_pairs_hh,
            rxn_specs=rxn_specs
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

        self.save_load_model(lyr, x_in)