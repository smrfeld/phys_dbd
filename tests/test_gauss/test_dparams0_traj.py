from physDBD import DParams0GaussTraj, DParams0Gauss
import numpy as np
import os
import tensorflow as tf

# Depreciation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class TestDParams0GaussTraj:

    def assert_equal_arrs(self, x_out, x_out_true):
        tol = 5.e-4
        assert np.max(abs(x_out-x_out_true)) < tol

    def create_dparams0_traj(self) -> DParams0GaussTraj:
        d1 = DParams0Gauss(
            nv=2,
            dmu_v=np.array([0.1,0.2]),
            dchol_v=np.array([[0.4,0.0],[0.8,0.9]])
            )
        d2 = DParams0Gauss(
            nv=2,
            dmu_v=np.array([0.4,0.7]),
            dchol_v=np.array([[0.8,0.0],[0.3,0.5]])
            )

        return DParams0GaussTraj(
            times=np.array([0.1,0.2]),
            dparams0_traj=[d1,d2]
            )        

    def test_nt(self):
        dt = self.create_dparams0_traj()
        assert dt.nt == 2

    def test_nv(self):
        dt = self.create_dparams0_traj()
        assert dt.nv == 2

    def test_get_tf_outputs(self):
        dt = self.create_dparams0_traj()
        outputs = dt.get_tf_outputs()
        
        self.assert_equal_arrs(outputs["dmu_v_0"], np.array([0.1,0.4]))
        self.assert_equal_arrs(outputs["dmu_v_1"], np.array([0.2,0.7]))
        self.assert_equal_arrs(outputs["dchol_v_0_0"], np.array([0.4,0.8]))
        self.assert_equal_arrs(outputs["dchol_v_1_0"], np.array([0.8,0.3]))
        self.assert_equal_arrs(outputs["dchol_v_1_1"], np.array([0.9,0.5]))

    def test_get_tf_outputs_normalized(self):
        dt = self.create_dparams0_traj()
        # Percent must be 1 to be deterministic
        outputs, mean, std_dev = dt.get_tf_outputs_normalized(percent=1.0)

        self.assert_equal_arrs(outputs["dmu_v_0"], np.array([-1.0,1.0]))
        self.assert_equal_arrs(outputs["dmu_v_1"], np.array([-1.0,1.0]))
        self.assert_equal_arrs(outputs["dchol_v_0_0"], np.array([-1.0,1.0]))
        self.assert_equal_arrs(outputs["dchol_v_1_0"], np.array([1.0,-1.0]))
        self.assert_equal_arrs(outputs["dchol_v_1_1"], np.array([1.0,-1.0]))

    def test_convert_to_pd(self):
        dt = self.create_dparams0_traj()
        df = dt.convert_to_pd()

        self.assert_equal_arrs(df["t"].to_numpy(), np.array([0.1,0.2]))
        self.assert_equal_arrs(df["dmu_v_0"].to_numpy(), np.array([0.1,0.4]))
        self.assert_equal_arrs(df["dmu_v_1"].to_numpy(), np.array([0.2,0.7]))
        self.assert_equal_arrs(df["dchol_v_0_0"].to_numpy(), np.array([0.4,0.8]))
        self.assert_equal_arrs(df["dchol_v_1_0"].to_numpy(), np.array([0.8,0.3]))
        self.assert_equal_arrs(df["dchol_v_1_1"].to_numpy(), np.array([0.9,0.5]))

    def test_export(self):
        dt = self.create_dparams0_traj()
        fname = "cache_params.txt"
        dt.export(fname)

        dt_back = DParams0GaussTraj.fromFile(fname,nv=2)
        assert dt == dt_back

        if os.path.exists(fname):
            os.remove(fname)

    def test_fromFile(self):
        dt = self.create_dparams0_traj()
        fname = "cache_params.txt"
        dt.export(fname)

        dt_back = DParams0GaussTraj.fromFile(fname,nv=2)
        assert dt == dt_back

        if os.path.exists(fname):
            os.remove(fname)
    
    def test_fromLFdict(self):
        lf_dict = {
            "dmu_v_0": np.array([0.1,0.4]),
            "dmu_v_1": np.array([0.2,0.7]),
            "dchol_v_0_0": np.array([0.4,0.8]),
            "dchol_v_1_0": np.array([0.8,0.3]),
            "dchol_v_1_1": np.array([0.9,0.5])
            }
        dt = DParams0GaussTraj.fromLFdict(
            times=np.array([0.1,0.2]),
            lf_dict=lf_dict,
            nv=2
            )

        dt1 = self.create_dparams0_traj()

        assert dt == dt1
