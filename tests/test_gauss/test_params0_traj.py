from physDBD import Params0Gauss, Params0GaussTraj, DParams0GaussTraj, DParams0Gauss
import numpy as np
import os
import tensorflow as tf

class TestParams0GaussTraj:

    nv = 2

    def assert_equal_arrs(self, x_out, x_out_true):
        tol = 5.e-4
        assert np.max(abs(x_out-x_out_true)) < tol

    def create_params_traj(self) -> Params0GaussTraj:
        p1 = Params0Gauss(
            nv=2,
            mu_v=np.array([1.0,2.0]),
            chol_v=np.array([[3.0,0.0],[4.0,5.0]])
            )
        p2 = Params0Gauss(
            nv=2,
            mu_v=np.array([1.1,4.0]),
            chol_v=np.array([[2.0,0.0],[6.0,7.0]])
            )
        p3 = Params0Gauss(
            nv=2,
            mu_v=np.array([1.4,3.0]),
            chol_v=np.array([[8.0,0.0],[8.0,9.0]])
            )

        return Params0GaussTraj(
            times=np.array([0.1,0.2,0.3]),
            params0_traj=[p1,p2,p3]
            )

    def test_fromIntegrating(self):
        times = np.array([0.1,0.2])
        d1 = DParams0Gauss(
            nv=2,
            dmu_v=np.array([0.1,0.2]),
            dchol_v=np.array([[0.1,0.0],[0.2,0.3]])
            )
        d2 = DParams0Gauss(
            nv=2,
            dmu_v=np.array([0.3,0.5]),
            dchol_v=np.array([[0.8,0.0],[0.7,0.9]])
            )
        dparams0_traj = DParams0GaussTraj(
            times=times,
            dparams0_traj=[d1,d2]
            )

        params0_init = Params0Gauss(
            nv=2,
            mu_v=np.array([0.5,0.1]),
            chol_v=np.array([[0.8,0.0],[0.4,0.9]])
            )
        
        pt = Params0GaussTraj.fromIntegrating(
            dparams0_traj=dparams0_traj,
            params0_init=params0_init,
            tpt_start=0,
            no_steps=2,
            constant_vals_lf={}
            )
        self.assert_equal_arrs(pt.params0_traj[0].mu_v, np.array([0.5,0.1]))
        self.assert_equal_arrs(pt.params0_traj[1].mu_v, np.array([0.6,0.3]))
        self.assert_equal_arrs(pt.params0_traj[2].mu_v, np.array([0.9,0.8]))

        self.assert_equal_arrs(pt.params0_traj[0].chol_v, np.array([[0.8,0.0],[0.4,0.9]]))
        self.assert_equal_arrs(pt.params0_traj[1].chol_v, np.array([[0.9,0.0],[0.6,1.2]]))
        self.assert_equal_arrs(pt.params0_traj[2].chol_v, np.array([[1.7,0.0],[1.3,2.1]]))
    
    def test_get_tf_inputs(self):
        pt = self.create_params_traj()
        non_zero_idx_pairs_vv = [(0,0),(1,0),(1,1)]
        inputs = pt.get_tf_inputs(non_zero_idx_pairs_vv)
        
        assert len(inputs["mu_v"]) == pt.nt-1
        assert len(inputs["chol_v_non_zero"]) == pt.nt-1
        for i in range(0,pt.nt-1):
            chol_v_non_zero = pt.params0_traj[i].chol_v_non_zero(non_zero_idx_pairs_vv)
            tf.debugging.assert_equal(
                tf.constant(pt.params0_traj[i].mu_v, dtype="float32"), 
                inputs["mu_v"][i].astype("float32")
                )
            tf.debugging.assert_equal(
                tf.constant(chol_v_non_zero, dtype="float32"), 
                inputs["chol_v_non_zero"][i].astype("float32")
                )
    
    def test_fromData(self):
        data1 = np.array([[3.0,5.0],[5.0,7.0]])
        data2 = np.array([[2.0,4.0],[4.0,6.0]])
        data = np.array([data1,data2])
        times = np.array([0.1,0.2])
        pt = Params0GaussTraj.fromData(data,times)
        self.assert_equal_arrs(
            pt.params0_traj[0].mu_v,
            np.array([4.0,6.0])
            )
        self.assert_equal_arrs(
            pt.params0_traj[1].mu_v,
            np.array([3.0,5.0])
            )

    def test_nt(self):
        pt = self.create_params_traj()
        assert pt.nt == 3

    def test_nv(self):
        pt = self.create_params_traj()
        assert pt.nv == 2

    def test_convert_to_pd(self):
        pt = self.create_params_traj()
        df = pt.convert_to_pd()

        self.assert_equal_arrs(df["t"].to_numpy(), np.array([0.1,0.2,0.3]))
        self.assert_equal_arrs(df["mu_v_0"].to_numpy(), np.array([1.0,1.1,1.4]))
        self.assert_equal_arrs(df["mu_v_1"].to_numpy(), np.array([2.0,4.0,3.0]))
        self.assert_equal_arrs(df["chol_v_0_0"].to_numpy(), np.array([3.0,2.0,8.0]))
        self.assert_equal_arrs(df["chol_v_1_0"].to_numpy(), np.array([4.0,6.0,8.0]))
        self.assert_equal_arrs(df["chol_v_1_1"].to_numpy(), np.array([5.0,7.0,9.0]))

    def test_export(self):
        pt = self.create_params_traj()
        fname = "cache_params.txt"
        pt.export(fname)

        pt_back = Params0GaussTraj.fromFile(fname,nv=self.nv)
        assert pt == pt_back

        if os.path.exists(fname):
            os.remove(fname)
    
    def test_fromFile(self):
        pt = self.create_params_traj()
        fname = "cache_params.txt"
        pt.export(fname)

        pt_back = Params0GaussTraj.fromFile(fname,nv=self.nv)
        assert pt == pt_back

        if os.path.exists(fname):
            os.remove(fname)

    def test_differentiate_with_TVR(self):
        alphas = {
            "mu_v_0": 1.0,
            "mu_v_1": 1.0,
            "chol_v_0_0": 1.0,
            "chol_v_1_0": 1.0,
            "chol_v_1_1": 1.0
        }

        pt = self.create_params_traj()
        dparams0_traj = pt.differentiate_with_TVR(
            alphas=alphas,
            no_opt_steps=10
            )