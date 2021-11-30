from physDBD import DataTypeGauss, TrainingGaussData, \
    Params0Gauss, Params0GaussTraj, DParams0Gauss, DParams0GaussTraj
import numpy as np
import os
import tensorflow as tf
import shutil

# Depreciation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class TestTrainingGaussData:

    def assert_equal_dicts(self, x_out, x_out_true):
        for key, val_true in x_out_true.items():
            val = x_out[key]

            self.assert_equal_arrs(val,val_true)

    def assert_equal_arrs(self, x_out, x_out_true):
        tol = 5.e-4
        assert np.max(abs(x_out-x_out_true)) < tol

    def create_params0_traj(self) -> Params0GaussTraj:
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
    
    def create_dparams0_traj(self) -> DParams0GaussTraj:
        p1 = DParams0Gauss(
            nv=2,
            dmu_v=np.array([0.3,0.4]),
            dchol_v=np.array([[0.3,0.0],[0.5,0.7]])
            )
        p2 = DParams0Gauss(
            nv=2,
            dmu_v=np.array([1.3,4.5]),
            dchol_v=np.array([[2.3,0.0],[6.6,7.7]])
            )

        return DParams0GaussTraj(
            times=np.array([0.1,0.2]),
            dparams0_traj=[p1,p2]
            )

    def test_reap_params0_traj_for_inputs(self):
        td = TrainingGaussData()
        pt = self.create_params0_traj()
        non_zero_idx_pairs_vv = [(0,0),(1,0),(1,1)]
        td.reap_params0_traj_for_inputs(
            params0_traj=pt, 
            data_type=DataTypeGauss.TRAINING,
            tpt_start_inc=0,
            tpt_end_exc=pt.nt-1,
            non_zero_idx_pairs_vv=non_zero_idx_pairs_vv
            )

        self.assert_equal_arrs(
            td.train_inputs['tpt'], np.array([0.0, 1.0]))
        self.assert_equal_arrs(
            td.train_inputs['mu_v'], np.array([[1.0, 2.0],[1.1, 4.0]]))
        self.assert_equal_arrs(
            td.train_inputs['chol_v_non_zero'], np.array([[3.0, 4.0, 5.0],[2.0, 6.0, 7.0]]))

    def test_reap_dparams0_traj_for_outputs(self):
        td = TrainingGaussData()
        dt = self.create_dparams0_traj()
        td.reap_dparams0_traj_for_outputs(
            dparams0_traj=dt, 
            tpt_start_inc=0,
            tpt_end_exc=dt.nt,
            data_type=DataTypeGauss.TRAINING
            )

        self.assert_equal_arrs(td.train_outputs_not_stdrd['dmu_v_0'], 
            np.array([0.3, 1.3]))
        self.assert_equal_arrs(td.train_outputs_not_stdrd['dmu_v_1'], 
            np.array([0.4, 4.5]))
        self.assert_equal_arrs(td.train_outputs_not_stdrd['dchol_v_0_0'], 
            np.array([0.3, 2.3]))
        self.assert_equal_arrs(td.train_outputs_not_stdrd['dchol_v_1_0'], 
            np.array([0.5, 6.6]))
        self.assert_equal_arrs(td.train_outputs_not_stdrd['dchol_v_1_1'], 
            np.array([0.7, 7.7]))

    def test_calculate_output_standardizations_and_apply(self):
        td = TrainingGaussData()
        dt = self.create_dparams0_traj()
        td.reap_dparams0_traj_for_outputs(
            dparams0_traj=dt, 
            tpt_start_inc=0,
            tpt_end_exc=dt.nt,
            data_type=DataTypeGauss.TRAINING
            )

        # Percent must be 1 to be deterministic
        td.calculate_output_standardizations_and_apply(percent=1.0)

        self.assert_equal_arrs(td.train_outputs_stdrd['dmu_v_0'], 
            np.array([-1.0, 1.0]))
        self.assert_equal_arrs(td.train_outputs_stdrd['dmu_v_1'], 
            np.array([-1.0, 1.0]))
        self.assert_equal_arrs(td.train_outputs_stdrd['dchol_v_0_0'], 
            np.array([-1.0, 1.0]))
        self.assert_equal_arrs(td.train_outputs_stdrd['dchol_v_1_0'], 
            np.array([-1.0, 1.0]))
        self.assert_equal_arrs(td.train_outputs_stdrd['dchol_v_1_1'], 
            np.array([-1.0, 1.0]))

    def write_read_standardizations(self, apply_on_read: bool):
        td = TrainingGaussData()
        dt = self.create_dparams0_traj()
        td.reap_dparams0_traj_for_outputs(
            dparams0_traj=dt,
            tpt_start_inc=0,
            tpt_end_exc=dt.nt,
            data_type=DataTypeGauss.TRAINING
            )

        # Percent must be 1 to be deterministic
        td.calculate_output_standardizations_and_apply(percent=1.0)

        dir_name = "data_stdrds"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        td.write_output_standardizations(dir_name)

        td2 = TrainingGaussData()
        dt = self.create_dparams0_traj()
        td2.reap_dparams0_traj_for_outputs(
            dparams0_traj=dt, 
            tpt_start_inc=0,
            tpt_end_exc=dt.nt,
            data_type=DataTypeGauss.TRAINING
            )

        if apply_on_read:
            td2.read_output_standardizations_and_apply(dir_name)
            self.assert_equal_dicts(td.train_outputs_stdrd, td2.train_outputs_stdrd)
        else:
            td2.read_output_standardizations(dir_name)
        self.assert_equal_dicts(td.train_outputs_mean, td2.train_outputs_mean)
        self.assert_equal_dicts(td.train_outputs_std_dev, td2.train_outputs_std_dev)

        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)

    def test_write_output_standardizations(self):
        self.write_read_standardizations(True)

    def test_read_output_standardizations(self):
        self.write_read_standardizations(False)

    def test_read_output_standardizations_and_apply(self):
        self.write_read_standardizations(True)
