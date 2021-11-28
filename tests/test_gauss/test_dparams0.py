from physDBD import DParams0Gauss
import numpy as np
import os
import tensorflow as tf

# Depreciation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class TestDParams0Gauss:

    def assert_equal_arrs(self, x_out, x_out_true):
        tol = 5.e-4
        assert np.max(abs(x_out-x_out_true)) < tol

    def test_get_tf_output(self):
        dp = DParams0Gauss(
            nv=2,
            dmu_v=np.array([0.2,0.3]),
            dchol_v=np.array([[0.5,0.0],[0.8,0.9]])
            )
        output = dp.get_tf_output()

        assert output["dmu_v_0"] == 0.2
        assert output["dmu_v_1"] == 0.3
        assert output["dchol_v_0_0"] == 0.5
        assert output["dchol_v_1_0"] == 0.8
        assert output["dchol_v_1_1"] == 0.9

    def test_to_lf_dict(self):
        dp = DParams0Gauss(
            nv=2,
            dmu_v=np.array([0.2,0.3]),
            dchol_v=np.array([[0.5,0.0],[0.8,0.9]])
            )
        lf_dict = dp.to_lf_dict()

        assert lf_dict["dmu_v_0"] == 0.2
        assert lf_dict["dmu_v_1"] == 0.3
        assert lf_dict["dchol_v_0_0"] == 0.5
        assert lf_dict["dchol_v_1_0"] == 0.8
        assert lf_dict["dchol_v_1_1"] == 0.9


    def test_fromLFdict(self):
        dp = DParams0Gauss(
            nv=2,
            dmu_v=np.array([0.2,0.3]),
            dchol_v=np.array([[0.5,0.0],[0.8,0.9]])
            )
        lf_dict = dp.to_lf_dict()

        dp1 = DParams0Gauss.fromLFdict(lf_dict,nv=2)
        assert dp == dp1

