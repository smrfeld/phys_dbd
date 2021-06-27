from physDBD import Params0Gauss, ImportHelper, Params0GaussTraj
import numpy as np
import os
import tensorflow as tf

class TestParams0Gauss:

    fnames = [
        "../data_test/0000.txt",
        "../data_test/0001.txt",
        "../data_test/0002.txt",
        "../data_test/0003.txt",
        "../data_test/0004.txt"
        ]
    species = ["ca2i","ip3"]
    nv = 2

    def import_params(self, time: float) -> Params0Gauss:
        data = ImportHelper.import_gillespie_ssa_at_time(
            fnames=self.fnames,
            time=time,
            species=self.species
        )

        params = Params0Gauss.fromData(data)
        return params

    def create_params_traj(self) -> Params0GaussTraj:
        return Params0GaussTraj(
            times=np.array([0.2,0.3,0.4,0.5,0.6,0.7]),
            params0_traj=[
                self.import_params(0.2),
                self.import_params(0.3),
                self.import_params(0.4),
                self.import_params(0.5),
                self.import_params(0.6),
                self.import_params(0.7)
                ]
            )

    def test_params(self):
        params = self.import_params(0.4)

    def test_export(self):
        pt = self.create_params_traj()
        fname = "cache_params.txt"
        pt.export(fname)

        # import back
        pt_back = Params0GaussTraj.fromFile(fname,nv=self.nv)

        # Check
        assert len(pt.params0_traj) == len(pt_back.params0_traj)
        for i in range(0,len(pt.params0_traj)):
            assert pt.params0_traj[i] == pt_back.params0_traj[i]

        if os.path.exists(fname):
            os.remove(fname)

    def test_tf_input(self):
        params = self.import_params(0.4)
        input0 = params.get_tf_input(tpt=0)
        
        tf.debugging.assert_equal(tf.constant(params.mu_v, dtype="float32"), input0["mu_v"].astype("float32"))
        tf.debugging.assert_equal(tf.constant(params.chol_v, dtype="float32"), input0["chol_v"].astype("float32"))

        pt = self.create_params_traj()
        inputs = pt.get_tf_inputs()
        
        assert len(inputs["mu_v"]) == len(pt.times)-1
        for i in range(0,len(inputs["mu_v"])):
            tf.debugging.assert_equal(tf.constant(pt.params0_traj[i].mu_v, dtype="float32"), inputs["mu_v"][i].astype("float32"))
            tf.debugging.assert_equal(tf.constant(pt.params0_traj[i].chol_v, dtype="float32"), inputs["chol_v"][i].astype("float32"))