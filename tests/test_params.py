from physPCA import Params, ImportHelper, ParamsTraj
import numpy as np
import os

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

    def test_params(self):
        params = self.import_params(0.4)

    def test_convert(self):
        params = self.import_params(0.4)
        arr = params.to_1d_arr()
        params_back = Params.from1dArr(arr, nv=2, nh=1)
        assert params == params_back

    def test_export(self):
        params1 = self.import_params(0.4)
        params2 = self.import_params(0.5)
        pt = ParamsTraj(
            times=np.array([0.4,0.5]),
            params_traj=[params1,params2]
            )
        pt.export_params_traj("cache.txt")

        # import back
        pt_back = ParamsTraj.fromFile("cache.txt",nv=2,nh=1)

        # Check
        assert len(pt.params_traj) == len(pt_back.params_traj)
        for i in range(0,len(pt.params_traj)):
            assert pt.params_traj[i] == pt_back.params_traj[i]

        if os.path.exists("cache.txt"):
            os.remove("cache.txt")

TestParams().test_params()