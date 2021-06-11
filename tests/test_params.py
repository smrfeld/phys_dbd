from physPCA import Params, ImportHelper, ParamsTraj, ParamsTETraj
import numpy as np
import os
import tensorflow as tf

class TestParams:

    fnames = [
        "data_test/0000.txt",
        "data_test/0001.txt",
        "data_test/0002.txt",
        "data_test/0003.txt",
        "data_test/0004.txt"
        ]
    species = ["ca2i","ip3"]

    def import_params(self, time: float) -> Params:
        data = ImportHelper.import_gillespie_ssa(
            fnames=self.fnames,
            time=time,
            species=self.species
        )

        muh = np.zeros(1)
        varh_diag = np.ones(1)
        params = Params.fromPCA(data,muh,varh_diag)
        return params

    def create_params_traj(self) -> ParamsTraj:
        return ParamsTraj(
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

    def test_convert(self):
        params = self.import_params(0.4)
        arr = params.to_1d_arr()
        params_back = Params.from1dArr(arr, nv=2, nh=1)
        assert params == params_back

    def test_export(self):
        pt = self.create_params_traj()
        fname = "cache_params.txt"
        pt.export(fname)

        # import back
        pt_back = ParamsTraj.fromFile(fname,nv=2,nh=1)

        # Check
        assert len(pt.params_traj) == len(pt_back.params_traj)
        for i in range(0,len(pt.params_traj)):
            assert pt.params_traj[i] == pt_back.params_traj[i]

        if os.path.exists(fname):
            os.remove(fname)

    def test_deriv(self):
        pt = self.create_params_traj()

        ptTE = pt.differentiate_with_TVR(
            alpha=1.0,
            no_opt_steps=10
        )

        # Export
        fname = "cache_deriv.txt"
        ptTE.export(fname)

        ptTE_back = ParamsTETraj.fromFile(fname,nv=2,nh=1)

        # Check
        assert len(ptTE.paramsTE_traj) == len(ptTE_back.paramsTE_traj)
        for i in range(0,len(ptTE.paramsTE_traj)):
            print("--- ", i)
            print(ptTE.paramsTE_traj[i])
            print(ptTE_back.paramsTE_traj[i])
            assert ptTE.paramsTE_traj[i] == ptTE_back.paramsTE_traj[i]

        if os.path.exists(fname):
            os.remove(fname)

    def test_tf_input(self):
        params = self.import_params(0.4)
        input0 = params.get_tf_input_assuming_params0()
        
        tf.debugging.assert_equal(tf.constant(params.wt, dtype="float32"), input0["wt"])
        tf.debugging.assert_equal(tf.constant(params.b, dtype="float32"), input0["b"])
        tf.debugging.assert_equal(tf.constant(params.sig2, dtype="float32"), input0["sig2"])

        pt = self.create_params_traj()
        inputs = pt.get_tf_inputs_assuming_params0()
        
        assert len(inputs) == len(pt.times)
        for i in range(0,len(inputs)):
            tf.debugging.assert_equal(tf.constant(pt.params_traj[i].wt, dtype="float32"), inputs[i]["wt"])
            tf.debugging.assert_equal(tf.constant(pt.params_traj[i].b, dtype="float32"), inputs[i]["b"])
            tf.debugging.assert_equal(tf.constant(pt.params_traj[i].sig2, dtype="float32"), inputs[i]["sig2"])