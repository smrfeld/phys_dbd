from physDBD import  ConvertMomentsToNMomentsLayer, DeathRxnLayer, \
    BirthRxnLayer, EatRxnLayer, ConvertNMomentsTEtoMomentsTE

# Depreciation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import copy
import os
import shutil

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

    _mu = np.array([19., 45., 62., 4., 8.])
    _cov = np.array([
        [30., 67., 107., 10., 9.], 
        [67., 162., 241., 20., 27.], 
        [107., 241., 402., 40., 27.], 
        [10., 20., 40., 5., 0.], 
        [9., 27., 27., 0., 9.]
        ])
    _ncov = np.array([
        [391., 922., 1285., 86., 161.],
        [922., 2187., 3031., 200., 387.],
        [1285., 3031., 4246., 288., 523.], 
        [86., 200., 288., 21., 32.], 
        [161., 387., 523., 32., 73.]
        ])

    _mu_TE = np.array([3.0, 5.0, 2.0, 1.0, 0.8])
    _ncov_TE = np.array([
        [12.0, 6.0, 3.0, 2.0, 1.0],
        [6.0, 18.0, 4.0, 3.0, 1.0],
        [3.0, 4.0, 16.0, 2.0, 1.0],
        [2.0, 3.0, 2.0, 8.0, 0.5],
        [1.0, 1.0, 1.0, 0.5, 6.0]
    ])
    
    @classmethod
    def mu_TE(cls):
        return np.tile(cls._mu_TE, (cls.batch_size,1))

    @classmethod
    def ncov_TE(cls):
        return np.tile(cls._ncov_TE, (cls.batch_size,1,1))

    @classmethod
    def mu(cls):
        return np.tile(cls._mu, (cls.batch_size,1))

    @classmethod
    def cov(cls):
        return np.tile(cls._cov, (cls.batch_size,1,1))

    @classmethod
    def ncov(cls):
        return np.tile(cls._ncov, (cls.batch_size,1,1))

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
        # Convert x_out_true covh to covh_diag as needed
        y_out_true = copy.copy(x_out_true)
        for key,val in x_out_true.items():
            if key == "covh":
                y_out_true["covh_diag"] = np.diag(val)
        if "covh" in y_out_true:
            del y_out_true["covh"]

        for key, val_true in y_out_true.items():
            val = x_out[key]

            self.assert_equal_arrs(val,val_true)

    def assert_equal_arrs(self, x_out, x_out_true):
        tol = 1.e-4
        assert np.max(abs(x_out-x_out_true)) < tol

    def save_load_model(self, lyr, x_in):

        # Test save; call the model once to build it first
        model = SingleLayerModel(lyr)
        x_out = model(x_in)

        model.save("saved_models/model", save_traces=False)

        # Test load
        model_rel = tf.keras.models.load_model("saved_models/model")
        print(model_rel)

        # Check types match!
        # Otherwise we may have: tensorflow.python.keras.saving.saved_model.load.XYZ instead of XYZ
        assert type(model_rel) is type(model)

        # Remove
        if os.path.isdir("saved_models"):
            shutil.rmtree("saved_models")

    def test_moments_to_nmoments(self):

        v = Vals()

        lyr = ConvertMomentsToNMomentsLayer()

        # Input
        x_in = {
            "mu": tf.constant(v.mu(), dtype="float32"),
            "cov": tf.constant(v.cov(), dtype="float32")
            }   
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "mu": np.array([19., 45., 62., 4., 8.]),
            "ncov": np.array([
                [391., 922., 1285., 86., 161.],
                [922., 2187., 3031., 200., 387.],
                [1285., 3031., 4246., 288., 523.], 
                [86., 200., 288., 21., 32.], 
                [161., 387., 523., 32., 73.]
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
            "ncov": tf.constant(v.ncov(), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "muTE": np.array([-19., 0., 0., 0., 0.]),
            "ncovTE": np.array([
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
            "ncov": tf.constant(v.ncov(), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "muTE": np.array([19., 0., 0., 0., 0.]),
            "ncovTE": np.array([
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
            "ncov": tf.constant(v.ncov(), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "muTE": np.array([-922., 922., 0., 0., 0.]),
            "ncovTE": np.array([
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
            "ncovTE": tf.constant(v.ncov_TE(), dtype="float32")
            }
            
        # Output
        x_out = lyr(x_in)

        print(x_out)

        x_out_true = {
            "muTE": np.array([3., 5., 2., 1., 0.8]),
            "covTE": np.array([
                [-102., -224., -221., -29., -38.2], 
                [-224., -432., -396., -62., -75.], 
                [-221., -396., -232., -68., -64.6], 
                [-29., -62., -68., 0., -10.7], 
                [-38.2, -75., -64.6, -10.7, -6.8]
            ])
        }

        self.assert_equal_dicts(x_out,x_out_true)

        self.save_load_model(lyr, x_in)