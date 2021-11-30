from physDBD import Params0Gauss, ImportHelper, DParams0Gauss
import numpy as np
import os
import tensorflow as tf

# Depreciation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class TestParams0Gauss:

    nv = 2

    def assert_equal_arrs(self, x_out, x_out_true):
        tol = 5.e-4
        assert np.max(abs(x_out-x_out_true)) < tol

    def import_params(self, time: float) -> Params0Gauss:
        fnames = [
            "../data_test/0000.txt",
            "../data_test/0001.txt",
            "../data_test/0002.txt",
            "../data_test/0003.txt",
            "../data_test/0004.txt"
            ]
        species = ["ca2i","ip3"]

        data = ImportHelper.import_gillespie_ssa_at_time(
            fnames=fnames,
            time=time,
            species=species
        )

        return Params0Gauss.fromData(data)

    def test_fromData(self):
        params = self.import_params(0.4)
        self.assert_equal_arrs(params.mu_v, np.array([698.2, 601.6]))
        self.assert_equal_arrs(params.chol_v, np.array([
            [ 0.03450852,  0.        ],
            [-0.12517618,  0.13210604]
            ]))

    def test_to_lf_dict(self):
        params = self.import_params(0.4)
        lf_dict = params.to_lf_dict()
        assert abs(lf_dict["mu_v_0"] - 698.2) < 1.e-6
        assert abs(lf_dict["mu_v_1"] - 601.6) < 1.e-6
        assert abs(lf_dict["chol_v_0_0"] - 0.034508516623074144) < 1.e-6
        assert abs(lf_dict["chol_v_1_0"] - -0.12517618115368168) < 1.e-6
        assert abs(lf_dict["chol_v_1_1"] - 0.13210604445218546) < 1.e-6

    def test_fromLFdict(self):
        params = self.import_params(0.4)
        lf_dict = params.to_lf_dict()
        params1 = Params0Gauss.fromLFdict(lf_dict, nv=self.nv)
        assert params == params1

    def test_nv(self):
        params = self.import_params(0.4)
        assert params.nv == self.nv

    def test_chol_v_non_zero(self):
        params = self.import_params(0.4)
        non_zero_idx_pairs_vv = [(0,0),(1,0),(1,1)]
        chol_v_non_zero = params.chol_v_non_zero(non_zero_idx_pairs_vv)
        self.assert_equal_arrs(
            chol_v_non_zero,
            np.array([0.03450852,  -0.12517618,  0.13210604])
            )

    def test_prec_v(self):
        params = self.import_params(0.4)
        self.assert_equal_arrs(params.prec_v, np.array([
            [0.00119084,   -0.00431964],
            [ -0.00431964,   0.03312108 ]
            ]))
    
    def test_cov_v(self):
        params = self.import_params(0.4)
        self.assert_equal_arrs(params.cov_v, np.array([
            [1593.7, 207.85],
            [207.85,  57.3]
            ]))

    def test_get_tf_input(self):
        params = self.import_params(0.4)
        non_zero_idx_pairs_vv = [(0,0),(1,0),(1,1)]
        input0 = params.get_tf_input(
            tpt=0,
            non_zero_idx_pairs_vv=non_zero_idx_pairs_vv
            )

        tf.debugging.assert_equal(
            tf.constant(params.mu_v, dtype="float32"), 
            input0["mu_v"].astype("float32")
            )
        chol_v_non_zero = params.chol_v_non_zero(non_zero_idx_pairs_vv)
        tf.debugging.assert_equal(
            tf.constant(chol_v_non_zero, dtype="float32"), 
            input0["chol_v_non_zero"].astype("float32")
            )
    
    def test_addParams0Gauss(self):
        params1 = self.import_params(0.4)
        params2 = self.import_params(0.4)
        params3 = Params0Gauss.addParams0Gauss(params1,params2)
        self.assert_equal_arrs(params3.mu_v, np.array([1396.4, 1203.2]))
        self.assert_equal_arrs(params3.chol_v, np.array([
            [ 0.06901703,  0.        ],
            [-0.25035236,  0.26421209]
            ]))

    def test_addLFdict(self):
        params = self.import_params(0.4)
        lf_dict = params.to_lf_dict()
        params3 = Params0Gauss.addLFdict(params,lf_dict)
        self.assert_equal_arrs(params3.mu_v, np.array([1396.4, 1203.2]))
        self.assert_equal_arrs(params3.chol_v, np.array([
            [ 0.06901703,  0.        ],
            [-0.25035236,  0.26421209]
            ]))

    def test_addDeriv(self):
        params = self.import_params(0.4)
        dparams = DParams0Gauss(
            nv=2,
            dmu_v=np.array([1.0,2.0]),
            dchol_v=np.array([
                [2.0,0.0],
                [4.0,8.0]
                ])
        )
        params1 = Params0Gauss.addDeriv(params,dparams)
        self.assert_equal_arrs(params1.mu_v, np.array([699.2, 603.6]))
        self.assert_equal_arrs(params1.chol_v, np.array([
            [ 2.03450852,  0.        ],
            [ 3.87482382,  8.13210604]
            ]))
