from helpers_test import SingleLayerModel
from physDBD import GGMmultPrecCovLayer, invert_ggm

import numpy as np
import tensorflow as tf

class TestGGMInv:

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

    def test_inv_prec_to_cov_mat(self):

        n = 2
        non_zero_idx_pairs = [(0,0),(1,0),(1,1)]

        # Create layer
        lyr = GGMmultPrecCovLayer.construct(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            init_diag_val=10.0
        )

        # Output
        x_in = {"cov_mat": np.array([[0.1,0.0],[0.0,0.1]])}
        x_out = lyr(x_in)

        print("Outputs: ",x_out)

        x_out_true = np.eye(n)

        self.assert_equal_arrs(x_out, x_out_true)

        self.save_load_model(lyr, x_in)

    def test_invert_ggm(self):

        n = 2
        non_zero_idx_pairs = [(0,0),(1,0),(1,1)]
        cov_mat = np.array([[0.1,0.0],[0.0,0.1]])

        prec_mat = invert_ggm(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            cov_mat=cov_mat
        )

        print(prec_mat)