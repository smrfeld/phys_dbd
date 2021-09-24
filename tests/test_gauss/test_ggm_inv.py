from helpers_test import SingleLayerModel
from physDBD import GGMmultPrecCovLayer, invert_ggm, invert_ggm_chol, construct_mat

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

    def test_invert_ggm_normal_n2(self):

        n = 2
        non_zero_idx_pairs = [(0,0),(1,0),(1,1)]
        cov_mat = np.array([[0.1,0.0],[0.0,0.1]])

        prec_mat_non_zero, final_loss = invert_ggm(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            cov_mat=cov_mat
        )

        print(final_loss)
        assert final_loss < 1.e-10

        prec_mat = construct_mat(n,non_zero_idx_pairs,prec_mat_non_zero)
        identity_check = np.dot(prec_mat,cov_mat)

        print(prec_mat)
        print(identity_check)

        self.assert_equal_arrs(identity_check, np.eye(n))

    def test_invert_ggm_normal_n3(self):

        n = 3
        non_zero_idx_pairs = [(0,0),(1,0),(1,1),(2,1),(2,2)]
        cov_mat = np.array([
            [10.0,5.0,2.0],
            [5.0,20.0,4.0],
            [2.0,4.0,30.0]
            ])

        prec_mat_non_zero, final_loss = invert_ggm(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            cov_mat=cov_mat,
            epochs=5000,
            learning_rate=0.001
        )

        print(final_loss)
        # assert final_loss < 1.e-10

        prec_mat = construct_mat(n,non_zero_idx_pairs,prec_mat_non_zero)
        print(prec_mat)
        print(np.linalg.inv(prec_mat))
        print("Should be")
        print(cov_mat)
        identity_check = np.dot(prec_mat,cov_mat)

        print(identity_check)

        self.assert_equal_arrs(identity_check, np.eye(n))

    def test_invert_ggm_chol(self):

        n = 2
        non_zero_idx_pairs = [(0,0),(1,0),(1,1)]
        cov_mat = np.array([[0.1,0.0],[0.0,0.1]])

        chol, final_loss = invert_ggm_chol(
            n=n,
            non_zero_idx_pairs=non_zero_idx_pairs,
            cov_mat=cov_mat
        )

        print(final_loss)
        print(chol)

        assert final_loss < 1.e-10
        # self.assert_equal_arrs(prec_mat, np.array([10.0,0.0,10.0]))