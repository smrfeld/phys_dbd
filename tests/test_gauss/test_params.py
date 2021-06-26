from physDBD import ParamsGauss, ImportHelper, ParamsGaussTraj
import numpy as np
import os
import tensorflow as tf

class TestParamsGauss:

    fnames = [
        "../data_test/0000.txt",
        "../data_test/0001.txt",
        "../data_test/0002.txt",
        "../data_test/0003.txt",
        "../data_test/0004.txt"
        ]
    species = ["ca2i","ip3"]
    nv = 2
    nh = 2

    def import_params(self, time: float) -> ParamsGauss:
        data = ImportHelper.import_gillespie_ssa_at_time(
            fnames=self.fnames,
            time=time,
            species=self.species
        )

        params = ParamsGauss.fromDataStd(data,self.nh)
        return params

    def create_params_traj(self) -> ParamsGaussTraj:
        return ParamsGaussTraj(
            times=np.array([0.2,0.3,0.4,0.5,0.6,0.7]),
            params_traj=[
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
        pt_back = ParamsGaussTraj.fromFile(fname,nv=self.nv,nh=self.nh)

        # Check
        assert len(pt.params_traj) == len(pt_back.params_traj)
        for i in range(0,len(pt.params_traj)):
            assert pt.params_traj[i] == pt_back.params_traj[i]

        if os.path.exists(fname):
            os.remove(fname)

    def test_tf_input(self):
        params = self.import_params(0.4)
        input0 = params.get_tf_input_assuming_params0(tpt=0)
        
        tf.debugging.assert_equal(tf.constant(params.mu, dtype="float32"), input0["mu"].astype("float32"))
        tf.debugging.assert_equal(tf.constant(params.chol, dtype="float32"), input0["chol"].astype("float32"))

        pt = self.create_params_traj()
        inputs = pt.get_tf_inputs_assuming_params0()
        
        assert len(inputs["mu"]) == len(pt.times)-1
        for i in range(0,len(inputs["mu"])):
            tf.debugging.assert_equal(tf.constant(pt.params_traj[i].mu, dtype="float32"), inputs["mu"][i].astype("float32"))
            tf.debugging.assert_equal(tf.constant(pt.params_traj[i].chol, dtype="float32"), inputs["chol"][i].astype("float32"))

    def test_convert(self):

        params = self.import_params(0.2)
        cov = params.cov
        cov_v = cov[:self.nv,:self.nv]

        mu_h_new = np.random.rand(self.nh)
        chol_vh_new = np.random.rand(self.nh,self.nv)
        chol_h_new = np.tril(np.random.rand(self.nh,self.nh))
        params.convert_latent_space(
            mu_h_new=mu_h_new,
            chol_vh_new=chol_vh_new,
            chol_h_new=chol_h_new
            )
        
        cov_new = params.cov
        cov_new_v = cov_new[:self.nv,:self.nv]

        assert np.max(abs(cov_v - cov_new_v)) < 1e-10